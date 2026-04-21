"""
AsyncRetriever for local_search_server_node.

Manages a pool of Ray actors and distributes requests via round-robin.
"""

import asyncio

import ray

from search_actors import BM25SearchActor, DenseSearchActor, PageAccessActor


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

        print(
            f"[AsyncRetriever] Initialized {len(self.search_actors)} search actors "
            f"(method={retrieval_method})."
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        num: int | None = None,
        return_score: bool = False,
    ):
        """Single-query search."""
        if num is None:
            num = self.topk
        try:
            async with self._search_sem:
                actor = await self._pick_search_actor()
                obj_ref = actor._search.remote(query, num, return_score)
                return await self._await_ray(obj_ref, self.search_timeout)
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
        """Batch search — dispatches the *entire* batch to a single actor
        (actors already run in parallel processes; splitting a batch across
        multiple actors would duplicate encode work)."""
        if num is None:
            num = self.topk
        try:
            async with self._search_sem:
                actor = await self._pick_search_actor()
                obj_ref = actor._batch_search.remote(query_list, num, return_score)
                return await self._await_ray(obj_ref, self.search_timeout)
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
