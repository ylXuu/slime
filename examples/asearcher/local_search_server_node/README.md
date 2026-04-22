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

- **BM25SearchActor** (`num_cpus=2, max_concurrency=64`) — Pyserini LuceneSearcher per actor
- **DenseSearchActor** (`num_gpus=1, num_cpus=2, max_concurrency=64`) — E5/BGE model on GPU + FAISS CPU index per actor
- **PageAccessActor** (`num_cpus=1, max_concurrency=256`) — single actor serving all `/access` requests

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

If you have **8 GPUs**, set `--num_search_actors 8 --faiss_gpu` so each actor gets one GPU.

### Recommended: 8-GPU remote node (high concurrency)

```bash
python -m local_search_server_node.server \
    --index_path /path/to/faiss.index \
    --corpus_path /path/to/corpus.jsonl \
    --pages_path /path/to/pages.jsonl \
    --retriever_name e5 \
    --retriever_model /path/to/e5-base-v2 \
    --num_search_actors 8 \
    --faiss_gpu \
    --max_search_concurrent 256 \
    --max_access_concurrent 256 \
    --search_batcher_size 64 \
    --search_batcher_wait_ms 10 \
    --port 8000
```

This configuration targets **≥2048 concurrent requests** from slime workers by:
- Spawning **8 Ray actors** (one per GPU) with `max_concurrency=64` each (512 total Ray-level concurrency)
- Enabling **FAISS GPU** to keep search on the same GPU as encode
- Enabling **Query Micro-Batching** (64 queries / 10 ms) so single-query requests are automatically aggregated into efficient GPU batches

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
| `--search_batcher_size` | `64` | Micro-batch size for aggregating single-query requests |
| `--search_batcher_wait_ms` | `10.0` | Max wait time (ms) to form a micro-batch |
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

## Remote Deployment & High Concurrency

When the search server runs on a **remote node** (not localhost), slime clients often see `Server disconnected` under high load. This is usually caused by:

1. **Firewall / NAT silently dropping idle keep-alive connections** after 60–300 s
2. **Single-query requests under-utilising GPU encode** (batch_size=1 vs sweet spot=64+)
3. **Uvicorn default backlog (128) being too small** for 2048+ concurrent clients

The following mitigations are built-in:

### Query Micro-Batching (`QueryBatcher`)

Because slime sends one query per HTTP request (`{"queries": [query]}`), GPU encode runs at batch=1, which is ~10× slower than batch=64.

`AsyncRetriever` now transparently aggregates single `search()` calls:
- Buffers incoming queries for up to `search_batcher_wait_ms` (default 10 ms)
- Flushes as soon as `search_batcher_size` (default 64) queries are collected
- Dispatches the merged batch via `_batch_search` to one Ray actor

For 2048 concurrent requests arriving in a burst, the server forms ~32 batches of 64 queries each and processes them in parallel across 8 actors, saturating all GPUs.

### Uvicorn tuning

```python
uvicorn.run(
    ...,
    loop="uvloop",
    timeout_keep_alive=300,   # > firewall idle timeout
    backlog=2048,             # allow more queued connections
)
```

- `uvloop` improves async I/O throughput
- `timeout_keep_alive=300` keeps HTTP keep-alive connections alive longer than typical NAT timeouts, reducing `Server disconnected`
- `backlog=2048` prevents SYN drops when thousands of clients connect simultaneously

### Actor concurrency limits

| Actor | max_concurrency | Purpose |
|-------|----------------|---------|
| BM25SearchActor | 64 | Enough for 8 actors × 64 = 512 concurrent BM25 searches |
| DenseSearchActor | 64 | 8 actors × 64 = 512 concurrent dense encode+search ops |
| PageAccessActor | 256 | Single actor handling all page lookups |

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
4. Tune `--search_batcher_size` and `--search_batcher_wait_ms`: larger batch = higher GPU util but slightly higher tail latency; shorter wait = lower latency but smaller batches

### `Server disconnected` (remote deployment)

This usually means the HTTP keep-alive connection was silently closed by a middlebox (firewall / NAT / cloud security group) while idle.

**On the server side** (already applied by default):
- `timeout_keep_alive=300` in uvicorn keeps connections alive longer than most NAT timeouts
- Ray actor `max_concurrency` increased so fewer requests queue at the HTTP layer

**On the slime client side** (if you choose to modify `search_clients.py`):
- Add `keepalive_timeout=30` to `TCPConnector` so aiohttp discards idle connections before NAT does
- Add `enable_cleanup_closed=True` to auto-purge dead connections
- Add one retry on `aiohttp.ServerDisconnectedError` for automatic failover to a fresh connection

## Reference

Inspired by [RAVine-server](https://github.com/ylXuu/RAVine-server) — a Ray-based async retriever with actor pools and round-robin scheduling.
