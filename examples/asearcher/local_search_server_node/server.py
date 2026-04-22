"""
Local Search Server Node — Ray-based multi-worker search engine.

Provides the same /retrieve and /access API as local_search_server.py,
but distributes work across multiple Ray actors for horizontal scaling.

Usage:
    # Single node, multi-GPU
    python -m local_search_server_node.server \
        --index_path /path/to/faiss.index \
        --corpus_path /path/to/corpus.jsonl \
        --pages_path /path/to/pages.jsonl \
        --retriever_name e5 \
        --retriever_model /path/to/e5-base-v2 \
        --topk 5 \
        --num_search_actors 4 \
        --port 8000

    # Then in generate_with_asearcher.py set:
    #   SEARCH_CONFIG["search_url"] = "http://<node_ip>:8000/retrieve"
    #   SEARCH_CONFIG["access_url"] = "http://<node_ip>:8000/access"
"""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from retriever import AsyncRetriever


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ServerConfig:
    def __init__(
        self,
        retrieval_method: str = "e5",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        pages_path: str | None = None,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = True,
        retrieval_batch_size: int = 512,
        faiss_gpu: bool = False,
        num_search_actors: int = 4,
        max_search_concurrent: int = 256,
        max_access_concurrent: int = 256,
        search_timeout: float = 60.0,
        access_timeout: float = 60.0,
        search_batcher_size: int = 64,
        search_batcher_wait_ms: float = 10.0,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.pages_path = pages_path
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.faiss_gpu = faiss_gpu
        self.num_search_actors = num_search_actors
        self.max_search_concurrent = max_search_concurrent
        self.max_access_concurrent = max_access_concurrent
        self.search_timeout = search_timeout
        self.access_timeout = access_timeout
        self.search_batcher_size = search_batcher_size
        self.search_batcher_wait_ms = search_batcher_wait_ms


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class AccessRequest(BaseModel):
    urls: list[str]


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

def create_lifespan(config: ServerConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("--- Local Search Server Node starting up... ---")
        print(f"Config: {vars(config)}")

        app.state.retriever = AsyncRetriever(
            retrieval_method=config.retrieval_method,
            index_path=config.index_path,
            corpus_path=config.corpus_path,
            model_path=config.retrieval_model_path,
            pages_path=config.pages_path,
            topk=config.retrieval_topk,
            num_search_actors=config.num_search_actors,
            max_search_concurrent=config.max_search_concurrent,
            max_access_concurrent=config.max_access_concurrent,
            search_timeout=config.search_timeout,
            access_timeout=config.access_timeout,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            batch_size=config.retrieval_batch_size,
            faiss_gpu=config.faiss_gpu,
            search_batcher_size=config.search_batcher_size,
            search_batcher_wait_ms=config.search_batcher_wait_ms,
        )

        print("--- Local Search Server Node startup complete. ---")
        yield
        print("--- Local Search Server Node shutting down... ---")

    return lifespan


# ---------------------------------------------------------------------------
# App factory (so we can pass config)
# ---------------------------------------------------------------------------

def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="Local Search Server Node (Ray)", lifespan=create_lifespan(config))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/retrieve")
    async def retrieve_endpoint(request: QueryRequest):
        """
        Dense/BM25 retrieval endpoint.

        Input:
            {"queries": ["..."], "topk": 5, "return_scores": false}
        Output:
            {"result": [[{"title": "...", "text": "...", "contents": "..."}, ...]]}
        """
        retriever: AsyncRetriever = app.state.retriever
        topk = request.topk if request.topk else retriever.topk

        # Single query -> fast path
        if len(request.queries) == 1:
            tmp = await retriever.search(
                request.queries[0], num=topk, return_score=request.return_scores
            )
        else:
            tmp = await retriever.batch_search(
                request.queries, num=topk, return_score=request.return_scores
            )

        scores = []
        try:
            results, scores = tmp
        except ValueError:
            results = tmp

        resp = []
        for i, single_result in enumerate(results):
            if scores:
                combined = []
                for doc, score in zip(single_result, scores[i], strict=True):
                    combined.append({"document": doc, "score": score})
                resp.append(combined)
            else:
                resp.append(single_result)
        return {"result": resp}

    @app.post("/access")
    async def access_endpoint(request: AccessRequest):
        """
        Lookup full page content by URL from the local pages dump.

        Input:
            {"urls": ["https://en.wikipedia.org/wiki/..."]}
        Output:
            {"result": [{"url": "...", "contents": "...", "title": "..."}, ...]}
        """
        retriever: AsyncRetriever = app.state.retriever
        pages = await retriever.access(request.urls)
        return {"result": pages}

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the ASearcher local search + access server (Ray multi-worker)."
    )
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index or BM25 Lucene index.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus jsonl file.")
    parser.add_argument("--pages_path", type=str, default=None, help="Path to pages dump jsonl (for /access endpoint).")
    parser.add_argument("--topk", type=int, default=5, help="Default number of retrieved passages.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Retriever name (e5, bge, dpr, bm25, ...).")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path to retriever model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for FAISS (one GPU per DenseSearchActor).")
    parser.add_argument("--num_search_actors", type=int, default=4, help="Number of Ray search actors (workers).")
    parser.add_argument("--max_search_concurrent", type=int, default=256, help="Max concurrent search requests.")
    parser.add_argument("--max_access_concurrent", type=int, default=256, help="Max concurrent access requests.")
    parser.add_argument("--search_timeout", type=float, default=60.0, help="Search timeout in seconds.")
    parser.add_argument("--access_timeout", type=float, default=60.0, help="Access timeout in seconds.")
    parser.add_argument("--search_batcher_size", type=int, default=64, help="Micro-batch size for single-query search aggregation.")
    parser.add_argument("--search_batcher_wait_ms", type=float, default=10.0, help="Max wait time (ms) to form a micro-batch.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray cluster address ('auto' or 'local').")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Init Ray
    # ------------------------------------------------------------------
    import ray

    if args.ray_address == "local":
        ray.init(ignore_reinit_error=True)
    else:
        try:
            ray.init(address=args.ray_address, ignore_reinit_error=True)
        except ConnectionError:
            print("[WARN] Could not connect to existing Ray cluster, starting local Ray...")
            ray.init(ignore_reinit_error=True)

    print(f"[Ray] Connected: {ray.cluster_resources()}")

    # ------------------------------------------------------------------
    # Build config & app
    # ------------------------------------------------------------------
    config = ServerConfig(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.topk,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        pages_path=args.pages_path,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        faiss_gpu=args.faiss_gpu,
        num_search_actors=args.num_search_actors,
        max_search_concurrent=args.max_search_concurrent,
        max_access_concurrent=args.max_access_concurrent,
        search_timeout=args.search_timeout,
        access_timeout=args.access_timeout,
        search_batcher_size=args.search_batcher_size,
        search_batcher_wait_ms=args.search_batcher_wait_ms,
    )

    app = create_app(config)
    print(f"[Server] Starting on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        loop="uvloop",
        timeout_keep_alive=300,
        backlog=2048,
    )
