"""
ASearcher generate function for slime.

Adopts Tongyi-DeepResearch's simple agent loop:
- Maintains conversation as a single text string.
- Detects <tool_call> / <answer> tags after each generation step.
- Supports two tools: search (local retrieval) and visit (page access).

Reference: examples/search-r1/generate_with_search.py
"""

import asyncio
import json
import re

from prompts import build_system_prompt
from search_clients import format_search_results, format_visit_result, local_search, local_visit

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_CONFIG = {
    "max_turns": 10,
    "topk": 5,
    "search_url": "http://127.0.0.1:8000/retrieve",
    "access_url": "http://127.0.0.1:8000/access",
    "return_logprob": True,
}


# ---------------------------------------------------------------------------
# Tool call parsing & execution
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """
    Parse <tool_call>{"name": "...", "arguments": {...}}</tool_call>.
    Returns (tool_name, arguments_dict) or None.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not name:
            return None
        return name, arguments
    except Exception:
        return None


async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a single tool call and return formatted result text."""
    if name == "search":
        query = arguments.get("query", "")
        if not query:
            return "Error: empty search query."
        results = await local_search(
            SEARCH_CONFIG["search_url"],
            query,
            top_k=SEARCH_CONFIG["topk"],
        )
        return format_search_results(results)

    if name == "visit":
        url = arguments.get("url", "")
        if not url:
            return "Error: empty visit URL."
        content = await local_visit(SEARCH_CONFIG["access_url"], url)
        return format_visit_result(content, url)

    return f"Error: unknown tool '{name}'. Available tools: search, visit."


async def process_turn(agent_text: str) -> tuple[str, bool]:
    """
    Inspect the latest assistant generation and decide next step.

    Returns:
        (observation_text, done)
        - observation_text: empty if done, else formatted tool response to append.
        - done: True if <answer> detected or no actionable tool call.
    """
    # Check for final answer
    if "<answer>" in agent_text and "</answer>" in agent_text:
        return "", True

    # Check for tool call
    parsed = parse_tool_call(agent_text)
    if parsed is None:
        # No valid tool call and no answer -> give a hint and continue
        return (
            "\nYou must either call a tool with <tool_call>...</tool_call> "
            "or provide a final answer with <answer>...</answer>.\n",
            False,
        )

    name, arguments = parsed
    tool_result = await execute_tool(name, arguments)
    obs = f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
    return obs, False


# ---------------------------------------------------------------------------
# Generate function (slime custom generate interface)
# ---------------------------------------------------------------------------

async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for asearcher at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Build initial prompt: system prompt + user question
    question = sample.prompt
    system_prompt = build_system_prompt()
    prompt_text = f"{system_prompt}\n\nUser: {question}\n\nAssistant: "
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Prepare accumulators
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_CONFIG["return_logprob"] else None

    # Stop sequences for tool_call and answer
    turn_sampling_params = dict(sampling_params)
    turn_sampling_params["stop"] = ["</tool_call>", "</answer>"]

    for _turn_idx in range(SEARCH_CONFIG["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": turn_sampling_params,
        }
        if SEARCH_CONFIG["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        # Extract tokens / logprobs
        if SEARCH_CONFIG["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found. Ensure 'return_logprob': True in payload."
                )
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
            cur_response_log_probs = []

        # Accumulate model-generated tokens (trainable)
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        if SEARCH_CONFIG["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        # If truncated by length, stop
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # Decide next step based on agent output
        obs_text, done = await process_turn(response)
        if done:
            break

        if obs_text:
            obs_token_ids = state.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
            response += obs_text
            response_token_ids += obs_token_ids
            loss_mask += [0] * len(obs_token_ids)
            if SEARCH_CONFIG["return_logprob"]:
                rollout_log_probs += [0.0] * len(obs_token_ids)
                assert len(response_token_ids) == len(rollout_log_probs), (
                    f"Token/logp mismatch: {len(response_token_ids)} vs {len(rollout_log_probs)}"
                )

    # Assemble final sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    if SEARCH_CONFIG["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED
        case _:
            sample.status = Sample.Status.COMPLETED

    return sample
