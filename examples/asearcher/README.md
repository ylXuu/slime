# ASearcher in slime (Local Retrieval)

This example adapts ASearcher to the slime framework, using a local Wikipedia corpus for retrieval and page access.

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Local server | `local_search_server.py` | FastAPI server with `/retrieve` (dense/BM25) and `/access` (page lookup) |
| Search clients | `search_clients.py` | Async wrappers for `/retrieve` and `/access` |
| Prompts | `prompts.py` | System prompt with `search` and `visit` tool definitions |
| Generate | `generate_with_asearcher.py` | Custom `generate()` for slime: multi-turn agent loop |
| Reward | `reward_func.py` | F1 + format reward + invalid penalty |
| Launch | `run_qwen3-4B-Instruct-2507.sh` | Training launch script |

## Setup

### 1. Prepare local retrieval index

You need three files:
- **Corpus** (`corpus.jsonl`): Wikipedia passages, one JSON per line with `title`, `text`, `contents`
- **FAISS index** (`e5_Flat.index`): Dense retrieval index built with e5/bge/etc.
- **Pages dump** (`pages.jsonl`): Full Wikipedia pages for `/access`, one JSON per line with `url`, `contents`

Example corpus entry:
```json
{"title": "Python (programming language)", "text": "Python is a high-level programming language...", "contents": "\"Python (programming language)\"\nPython is a high-level programming language..."}
```

Example pages entry:
```json
{"url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "contents": "Python is a high-level..."}
```

### 2. Start local search server

```bash
python local_search_server.py \
    --index_path /path/to/e5_Flat.index \
    --corpus_path /path/to/corpus.jsonl \
    --pages_path /path/to/pages.jsonl \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --topk 5 \
    --port 8000
```

Test endpoints:
```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Python programming language"], "topk": 3}'

curl -X POST http://127.0.0.1:8000/access \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://en.wikipedia.org/wiki/Python_(programming_language)"]}'
```

### 3. Prepare training data

ASearcher uses JSONL with `question` and `answer` fields. Convert to slime format:

```python
import json

with open("asearcher_train.jsonl", "w") as fout:
    for item in original_data:
        fout.write(json.dumps({
            "prompt": item["question"],
            "answer": item["answer"]  # string or list of strings
        }) + "\n")
```

Save as `.jsonl` or convert to `.parquet`.

### 4. Launch training

Edit `run_qwen3-4B-Instruct-2507.sh` to set:
- `--hf-checkpoint`: path to your Qwen3-4B-Instruct-2507 checkpoint
- `--prompt-data`: path to your training data

Then run:
```bash
bash run_qwen3-4B-Instruct-2507.sh
```

## Agent Behavior

The agent follows a simple ReAct loop:

1. **System prompt + user question** are fed to the model.
2. The model generates until `</tool_call>` or `</answer>`.
3. If `<tool_call>` is detected:
   - `search`: query the local corpus, return top-k passages.
   - `visit`: fetch a full page by URL.
   - Result is appended as `<tool_response>...</tool_response>`.
   - Loop continues.
4. If `<answer>` is detected, the episode ends.

## Reward

- **F1 score**: Chinese-aware token-level F1 between predicted and ground-truth answers.
- **Format reward**: 1.0 if all `<search>`, `<access>`, `<answer>` tags are balanced; 0.0 otherwise.
- **Invalid penalty**: -0.5 if the agent incorrectly claims a valid question is invalid.

Final reward = `f1_score * format_reward` (plus possible invalid penalty).

## Customization

| Config key in `generate_with_asearcher.py` | Default | Description |
|-------------------------------------------|---------|-------------|
| `SEARCH_CONFIG["max_turns"]` | 10 | Max agent turns per episode |
| `SEARCH_CONFIG["topk"]` | 5 | Retrieved passages per search |
| `SEARCH_CONFIG["search_url"]` | `http://127.0.0.1:8000/retrieve` | Local retrieve endpoint |
| `SEARCH_CONFIG["access_url"]` | `http://127.0.0.1:8000/access` | Local access endpoint |
| `SEARCH_CONFIG["return_logprob"]` | `True` | Collect logprobs for TIS |

## References

- ASearcher: https://github.com/inclusionAI/ASearcher
- Tongyi-DeepResearch: https://github.com/Alibaba-NLP/DeepResearch
- slime search-r1 example: `../search-r1/`
