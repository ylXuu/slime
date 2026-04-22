"""
AsyncRetriever for local_search_server_node.

Manages a pool of Ray actors and distributes requests via round-robin.
Implements micro-batching for single-query requests to saturate GPU encode.
"""

import asyncio

import ray

from search_actors import BM25SearchActor, DenseSearchActor, PageAccessActor


# ---------------------------------------------------------------------------
# Query Batcher — aggregates single-query requests into batches for GPU efficiency
# ---------------------------------------------------------------------------

class QueryBatcher:
    """
    Async micro-batcher that collects single-query search requests over a short
    time window and dispatches them as a single batch to one Ray actor.

    This dramatically increases GPU utilisation because the encoder's sweet spot
    is batch_size >= 64, whereas slime currently sends queries one-by-one.
    """

    def __init__(
        self,
        retriever,
        max_batch_size: int = 64,
        max_wait_ms: float = 10.0,
    ):
        self.retriever = retriever
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def submit(
        self,
        query: str,
        num: int,
        return_score: bool,
    ):
        """Submit a single query and await its result."""
        if self._closed:
            raise RuntimeError("QueryBatcher is closed")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._dispatch_loop())

        fut = asyncio.get_event_loop().create_future()
        await self._queue.put((query, num, return_score, fut))
        return await fut

    async def close(self):
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _dispatch_loop(self):
        while not self._closed:
            batch = []
            # Block until at least one item arrives
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self.max_wait_ms / 1000.0
                )
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Pull as many more items as we can without blocking
            while len(batch) < self.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self._flush(batch)

    async def _flush(self, batch: list):
        queries = [item[0] for item in batch]
        num = batch[0][1]
        return_score = batch[0][2]

        try:
            results, scores = await self.retriever._raw_batch_search(
                queries, num, return_score
            )
            for i, (_, _, _, fut) in enumerate(batch):
                if not fut.done():
                    if return_score:
                        fut.set_result((results[i], scores[i]))
                    else:
                        fut.set_result(results[i])
        except Exception as exc:
            for _, _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)


# ---------------------------------------------------------------------------
# Async Retriever
# ---------------------------------------------------------------------------

class AsyncRetriever:
    """
    High-level async retriever that dispatches requests to a pool of Ray actors.
    """

    def __init__(
        self,
        retrieval_method: str,
        index_path: str,
        corpus_path: str,
        model_path: str | None = None,
        pages_path: str | None = None,
        topk: int = 5,
        num_search_actors: int = 4,
        max_search_concurrent: int = 256,
        max_access_concurrent: int = 256,
        search_timeout: float = 60.0,
        access_timeout: float = 60.0,
        pooling_method: str = "mean",
        max_length: int = 256,
        use_fp16: bool = True,
        batch_size: int = 512,
        faiss_gpu: bool = False,
        # Batcher config
        search_batcher_size: int = 64,
        search_batcher_wait_ms: float = 10.0,
    ):
        self.retrieval_method = retrieval_method
        self.topk = topk
        self.search_timeout = search_timeout
        self.access_timeout = access_timeout

        # ------------------------------------------------------------------
        # Create search actors
        # ------------------------------------------------------------------
        if retrieval_method == "bm25":
            self.search_actors = [
                BM25SearchActor.remote(
                    index_path=index_path,
                    corpus_path=corpus_path,
                    topk=topk,
                )
                for _ in range(num_search_actors)
            ]
        else:
            assert model_path is not None, "model_path is required for dense retriever"
            self.search_actors = [
                DenseSearchActor.remote(
                    index_path=index_path,
                    corpus_path=corpus_path,
                    retrieval_method=retrieval_method,
                    model_path=model_path,
                    pooling_method=pooling_method,
                    max_length=max_length,
                    use_fp16=use_fp16,
                    topk=topk,
                    batch_size=batch_size,
                    faiss_gpu=faiss_gpu,
                )
                for _ in range(num_search_actors)
            ]

        # ------------------------------------------------------------------
        # Create page access actor (single actor is enough, fetch is cheap)
        # ------------------------------------------------------------------
        self.page_access_actor = (
            PageAccessActor.remote(pages_path) if pages_path else None
        )

        # ------------------------------------------------------------------
        # Round-robin state & semaphores
        # ------------------------------------------------------------------
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()
        self._search_sem = asyncio.Semaphore(max_search_concurrent)
        self._access_sem = asyncio.Semaphore(max_access_concurrent)

        # ------------------------------------------------------------------
        # Query batcher for single-query saturation
        # ------------------------------------------------------------------
        self._batcher = QueryBatcher(
            self,
            max_batch_size=search_batcher_size,
            max_wait_ms=search_batcher_wait_ms,
        )

        print(
            f"[AsyncRetriever] Initialized {len(self.search_actors)} search actors "
            f"(method={retrieval_method}, batcher_size={search_batcher_size}, "
            f"batcher_wait={search_batcher_wait_ms}ms)."
        )
        if self.page_access_actor:
            print("[AsyncRetriever] Page access actor initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _pick_search_actor(self):
        async with self._rr_lock:
            actor = self.search_actors[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.search_actors)
            return actor

    @staticmethod
    async def _await_ray(obj_ref, timeout: float):
        """Await a Ray ObjectRef with asyncio timeout."""
        return await asyncio.wait_for(obj_ref, timeout=timeout)

    async def _raw_batch_search(
        self,
        query_list: list[str],
        num: int,
        return_score: bool,
    ):
        """Direct batch search to a Ray actor (used by QueryBatcher and batch_search)."""
        async with self._search_sem:
            actor = await self._pick_search_actor()
            obj_ref = actor._batch_search.remote(query_list, num, return_score)
            return await self._await_ray(obj_ref, self.search_timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        num: int | None = None,
        return_score: bool = False,
    ):
        """Single-query search — transparently micro-batched for GPU efficiency."""
        if num is None:
            num = self.topk
        try:
            return await self._batcher.submit(query, num, return_score)
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] search timeout after {self.search_timeout}s: {query}")
            return ([], []) if return_score else []
        except Exception as e:
            print(f"[AsyncRetriever] search error: {e}")
            return ([], []) if return_score else []

    async def batch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ):
        """Batch search — already a batch, dispatch directly."""
        if num is None:
            num = self.topk
        try:
            return await self._raw_batch_search(query_list, num, return_score)
        except asyncio.TimeoutError:
            print(
                f"[AsyncRetriever] batch_search timeout after {self.search_timeout}s, "
                f"queries={len(query_list)}"
            )
            return ([], []) if return_score else []
        except Exception as e:
            print(f"[AsyncRetriever] batch_search error: {e}")
            return ([], []) if return_score else []

    async def access(self, urls: list[str]):
        """Batch page access."""
        if self.page_access_actor is None:
            return [{"url": url, "contents": "", "error": "No pages loaded"} for url in urls]
        try:
            async with self._access_sem:
                obj_ref = self.page_access_actor.access.remote(urls)
                return await self._await_ray(obj_ref, self.access_timeout)
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] access timeout after {self.access_timeout}s, urls={len(urls)}")
            return [{"url": url, "contents": "", "error": "Timeout"} for url in urls]
        except Exception as e:
            print(f"[AsyncRetriever] access error: {e}")
            return [{"url": url, "contents": "", "error": str(e)} for url in urls]
