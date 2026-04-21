# Local Search Server Node (Ray Multi-Worker)

A horizontally-scalable replacement for `local_search_server.py` that distributes search and access requests across multiple **Ray actors**.

## Why?

The original `local_search_server.py` runs as a single process. Under high concurrency (e.g. 256 samples × multi-turn agent loop), the single-process server becomes the bottleneck:

- `/retrieve` is a **sync** FastAPI endpoint → limited by the default thread pool (~32 threads)
- Each request loads the model, encodes queries on GPU, and searches FAISS serially
- `/access` gets starved when `/retrieve` hogs the thread pool

This node solves the problem by:

1. **Creating N independent Ray actors**, each with its own model + index copy
2. **Round-robin dispatching** requests across actors
3. **Async endpoints** (`async def`) so FastAPI never blocks the event loop
4. **Separate semaphores** for search vs access, preventing starvation

## Architecture

```
Client (slime)          FastAPI Server          Ray Actors
     |                        |                        |
     |-- POST /retrieve ----->|                        |
     |                        |-- pick actor (RR) ---->|
     |                        |                        |-- GPU encode
     |                        |                        |-- CPU FAISS search
     |<-- JSON result --------|<-- ObjectRef result ---|
     |                        |                        |
     |-- POST /access -------->|                        |
     |                        |-- PageAccessActor ---->|
     |                        |                        |-- dict lookup
     |<-- JSON result --------|<-- ObjectRef result ---|
```

- **BM25SearchActor** (`num_cpus=2, max_concurrency=20`) — Pyserini LuceneSearcher per actor
- **DenseSearchActor** (`num_gpus=1, num_cpus=2, max_concurrency=20`) — E5/BGE model on GPU + FAISS CPU index per actor
- **PageAccessActor** (`num_cpus=1, max_concurrency=50`) — single actor serving all `/access` requests

## Quick Start

### 1. Start the server (single node, multi-GPU)

```bash
cd examples/asearcher

python -m local_search_server_node.server \
    --index_path /cpfs01/yilong.xu/datasets/wiki-18-e5-index/e5_Flat.index \
    --corpus_path /cpfs01/yilong.xu/datasets/ASearcher-Local-Knowledge/wiki_corpus.jsonl \
    --pages_path /cpfs01/yilong.xu/datasets/ASearcher-Local-Knowledge/wiki_webpages.jsonl \
    --retriever_name e5 \
    --retriever_model /cpfs01/yilong.xu/models/e5-base-v2 \
    --topk 5 \
    --num_search_actors 4 \
    --max_search_concurrent 256 \
    --max_access_concurrent 256 \
    --port 8000
```

If you have **8 GPUs**, set `--num_search_actors 8` so each actor gets one GPU.

### 2. Update slime script to point to the new server

In `run_qwen3-4B-Instruct-2507.sh` (or your training script), update the URLs in `generate_with_asearcher.py`:

```python
SEARCH_CONFIG = {
    "max_turns": 30,
    "topk": 5,
    "search_url": "http://<server_node_ip>:8000/retrieve",
    "access_url": "http://<server_node_ip>:8000/access",
    ...
}
```

Or set via environment variables and read them in the generate function.

### 3. Connect to an existing Ray cluster (optional)

```bash
# If you already have a Ray cluster running
python -m local_search_server_node.server \
    ... \
    --ray_address auto   # or "ray://head-node-ip:10001"
```

Default is `auto` which tries to connect to an existing cluster; falls back to `local` if none exists.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--index_path` | **required** | Path to FAISS index or BM25 Lucene index |
| `--corpus_path` | **required** | Path to corpus jsonl file |
| `--pages_path` | `None` | Path to pages dump jsonl (for `/access`) |
| `--retriever_name` | `e5` | Retriever name: `e5`, `bge`, `dpr`, `bm25` |
| `--retriever_model` | `intfloat/e5-base-v2` | HuggingFace model path or local path |
| `--num_search_actors` | `4` | Number of Ray search worker actors |
| `--max_search_concurrent` | `256` | asyncio.Semaphore for `/retrieve` |
| `--max_access_concurrent` | `256` | asyncio.Semaphore for `/access` |
| `--search_timeout` | `60.0` | Timeout per search request (seconds) |
| `--access_timeout` | `60.0` | Timeout per access request (seconds) |
| `--port` | `8000` | FastAPI server port |
| `--host` | `0.0.0.0` | FastAPI server host |
| `--ray_address` | `auto` | Ray cluster address |

## API Compatibility

This server is **drop-in compatible** with `local_search_server.py`:

### `POST /retrieve`

**Request:**
```json
{
  "queries": ["Who is Ruth Scurr?"],
  "topk": 5,
  "return_scores": false
}
```

**Response:**
```json
{
  "result": [
    [
      {"title": "Ruth Scurr", "text": "...", "contents": "...", "url": "..."}
    ]
  ]
}
```

### `POST /access`

**Request:**
```json
{
  "urls": ["https://en.wikipedia.org/wiki/Ruth_Scurr"]
}
```

**Response:**
```json
{
  "result": [
    {"url": "https://en.wikipedia.org/wiki/Ruth_Scurr", "contents": "...", "title": "..."}
  ]
}
```

## Design Notes

### Why not share the model/index across actors?

Ray actors are **separate processes**. Sharing a PyTorch model or FAISS index between processes requires complex shared-memory tricks (e.g. `ray.put` + zero-copy plasma store), which breaks easily with GPU tensors and FAISS GPU indexes.

Instead, each actor loads its **own copy** of the (small) E5 model and FAISS **CPU** index. For E5-base-v2 (~100 MB) and a Flat CPU index, the memory overhead is acceptable. The throughput gain from N parallel GPU encodes far outweighs the memory cost.

### Why CPU FAISS instead of GPU FAISS?

- `faiss.index_cpu_to_all_gpus()` would pin the entire index to **all** GPUs, which conflicts with `num_gpus=1` per actor
- Flat CPU index search for `topk=5` is sub-millisecond; the bottleneck is **GPU encode**
- If you need GPU FAISS, consider loading a shard per actor (advanced)

### What about `/access`?

`/access` only does dict lookups (microseconds). A single `PageAccessActor` with `max_concurrency=50` is more than enough. The `asyncio.Semaphore` in `AsyncRetriever` protects against thundering herd.

## Troubleshooting

### `RuntimeError: No CUDA GPUs are available`

Make sure the machine running this server has GPUs. If you want CPU-only mode, you can modify `search_actors.py` to remove `num_gpus=1` from `DenseSearchActor`, but encode will be very slow.

### `ConnectionError: Could not connect to Ray cluster`

The server falls back to `ray.init()` (local mode). If you see this warning, it just means no existing cluster was found; a local one is started automatically.

### High latency under load

1. Increase `--num_search_actors` to match your GPU count
2. Increase `--max_search_concurrent` if you have enough actor capacity
3. Check GPU utilization with `nvidia-smi dmon`; if GPUs are under-utilized, increase actor count

## Reference

Inspired by [RAVine-server](https://github.com/ylXuu/RAVine-server) — a Ray-based async retriever with actor pools and round-robin scheduling.
