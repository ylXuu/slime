"""
ASearcher reward function for slime.

Ports reward logic from ASearcher/utils/rewards.py:
- Chinese-aware F1 score
- Format reward (tag matching)
- Invalid question penalty (-0.5)
"""

import re
import string


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    return s


def contains_chinese(text: str) -> bool:
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3400" <= char <= "\u4dbf":
            return True
        if "\uf900" <= char <= "\ufaff":
            return True
    return False


def normalize_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_solution(solution_str: str) -> str | None:
    """Extract the last <answer>...</answer> content."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def em_check(prediction: str, golden_answers: list[str]) -> int:
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    for golden_answer in golden_answers:
        if normalize_answer(bool_mapping(golden_answer)) == normalized_prediction:
            return 1
    return 0


def f1_score(answer_content: str, gt: str) -> float:
    answer_content = normalize_text(bool_mapping(answer_content))
    gt = normalize_text(bool_mapping(gt))

    if contains_chinese(gt):
        def parse_chinese_str(s):
            numbers = []
            for i, c in enumerate(s):
                if c.isdigit():
                    if i > 0 and s[i - 1].isdigit():
                        numbers[-1] = numbers[-1] + c
                    else:
                        numbers.append(c)
            for c in "0123456789，。 ,.-":
                s = s.replace(c, "")
            return set(list(s) + numbers)

        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(gt)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt.split())

    if not gt_tokens or not pred_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_f1(solution_str: str, ground_truth) -> tuple[str | None, float]:
    """
    Compute F1 score. ground_truth can be a single string or a list of candidates.
    Returns (extracted_answer, f1_score).
    """
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str)
        if answer is None:
            return None, 0.0
        scores = [f1_score(answer, g) for g in ground_truth]
        return answer, max(scores)

    answer = extract_solution(solution_str)
    if answer is None:
        return None, 0.0
    return answer, f1_score(answer, ground_truth)


def compute_em(solution_str: str, ground_truth) -> tuple[str | None, int]:
    """Compute exact match. Returns (extracted_answer, 0_or_1)."""
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str)
        if answer is None:
            return None, 0
        return answer, em_check(answer, ground_truth)

    answer = extract_solution(solution_str)
    if answer is None:
        return None, 0
    return answer, em_check(answer, [ground_truth])


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

def correct_format(s: str) -> bool:
    """Check if a single turn has correct tag balance."""
    return all(
        [
            s.count("<search>") == s.count("</search>"),
            s.count("<access>") == s.count("</access>"),
            s.count("<answer>") == s.count("</answer>"),
            s.count("<search>") + s.count("<access>") + s.count("<answer>") <= 1,
            s.count("Assistant") == s.count("assistant") == 0,
            s.count("</think>") <= 1,
        ]
    )


def compute_format_reward(full_text: str) -> float:
    """
    Check format correctness across the whole trajectory.
    We look at the final assistant response (after the last 'Assistant:').
    """
    # Split by assistant markers to get each generation turn
    turns = re.split(r"Assistant:\s*", full_text)
    if len(turns) <= 1:
        # No assistant generation found
        return 0.0

    # Check every assistant turn for correct format
    # The first item in turns is the prompt, skip it
    for turn in turns[1:]:
        # Each turn may end with tool_response or be the final answer
        # We only check the assistant-generated part (before any tool_response)
        gen_part = turn.split("<tool_response>")[0] if "<tool_response>" in turn else turn
        if not correct_format(gen_part):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Main reward function (slime interface)
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs) -> float:
    """
    ASearcher reward function.

    Args:
        args: CLI arguments
        sample: slime Sample object

    Returns:
        float reward
    """
    if not hasattr(sample, "prompt") or not hasattr(sample, "response"):
        return 0.0

    full_text = sample.prompt + sample.response
    ground_truth = sample.label

    # Handle ground_truth format: may be string or list
    if ground_truth is None:
        return 0.0

    # Compute F1
    answer, f1 = compute_f1(full_text, ground_truth)

    # Format reward
    format_reward = compute_format_reward(full_text)

    # Compose base score
    score = f1 * format_reward

    # Invalid question penalty
    # If the agent claims a valid question is invalid, penalize
    if answer is not None:
        invalid_keywords = ["question is invalid", "invalid", "appropriate", "valid"]
        lower_answer = answer.lower()
        if any(kw in lower_answer for kw in invalid_keywords):
            # Check if the ground truth suggests this is a real question
            # (If ground truth is not empty/not "invalid", then it's a valid question)
            gt_text = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            gt_text = str(gt_text).lower()
            if gt_text and "invalid" not in gt_text:
                score = -0.5

    return float(score)
