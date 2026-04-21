"""
Ray Actor definitions for local_search_server_node.

Each actor is an independent worker process that loads its own model + index.
Requests are distributed across actors via round-robin by AsyncRetriever.
"""

import json
import os
import warnings

import faiss
import numpy as np
import ray
import torch
from transformers import AutoModel, AutoTokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class _Encoder:
    """Internal encoder helper (loaded inside each actor process)."""

    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.model.cuda()
        if use_fp16:
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [f"{prefix}{q}" for q in query_list]

        if "bge" in self.model_name.lower() and is_query:
            query_list = [
                f"Represent this sentence for searching relevant passages: {q}"
                for q in query_list
            ]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].shape[0], 1), dtype=torch.long
            ).to(inputs["input_ids"].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        return query_emb


# ---------------------------------------------------------------------------
# Base Search Actor
# ---------------------------------------------------------------------------

class BaseSearchActor:
    """Base class for search actors (not a Ray actor itself)."""

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# BM25 Search Actor (Ray)
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=2, max_concurrency=20)
class BM25SearchActor(BaseSearchActor):
    """Ray actor that wraps a BM25 (Pyserini Lucene) searcher."""

    def __init__(
        self,
        index_path: str,
        corpus_path: str | None,
        topk: int = 5,
    ):
        from pyserini.search.lucene import LuceneSearcher

        self.index_path = index_path
        self.corpus_path = corpus_path
        self.topk = topk

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()

        if not self.contain_doc and self.corpus_path:
            self.corpus_list = self._load_corpus_list(self.corpus_path)
        else:
            self.corpus_list = []

    def _check_contain_doc(self):
        try:
            return self.searcher.doc(0).raw() is not None
        except Exception:
            return False

    @staticmethod
    def _load_corpus_list(corpus_path: str) -> list[dict]:
        import datasets

        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
        return [corpus[i] for i in range(len(corpus))]

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            return ([], []) if return_score else []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_raw = [json.loads(self.searcher.doc(hit.docid).raw()) for hit in hits]
            results = [
                {
                    "title": raw.get("contents", "").split("\n")[0].strip('"'),
                    "text": "\n".join(raw.get("contents", "").split("\n")[1:]),
                    "contents": raw.get("contents", ""),
                    "url": raw.get("url", ""),
                }
                for raw in all_raw
            ]
        else:
            results = [self.corpus_list[int(hit.docid)] for hit in hits]

        return (results, scores) if return_score else results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        return (results, scores) if return_score else results


# ---------------------------------------------------------------------------
# Dense Search Actor (Ray) — GPU encode + CPU FAISS
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1, num_cpus=2, max_concurrency=20)
class DenseSearchActor(BaseSearchActor):
    """Ray actor that wraps a dense retriever (E5/BGE/DPR + FAISS CPU)."""

    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        retrieval_method: str,
        model_path: str,
        pooling_method: str = "mean",
        max_length: int = 256,
        use_fp16: bool = True,
        topk: int = 5,
        batch_size: int = 512,
        faiss_gpu: bool = False,
    ):
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.retrieval_method = retrieval_method
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.topk = topk
        self.batch_size = batch_size

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            # FP16 can cause instability for some models (e.g. E5), disable by default
            co.useFloat16 = False
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # Pre-load corpus into memory
        self.corpus_list = self._load_corpus_list(self.corpus_path)

        # Load encoder
        self.encoder = _Encoder(
            model_name=self.retrieval_method,
            model_path=self.model_path,
            pooling_method=self.pooling_method,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
        )

    @staticmethod
    def _load_corpus_list(corpus_path: str) -> list[dict]:
        import datasets

        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
        return [corpus[i] for i in range(len(corpus))]

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = [self.corpus_list[int(i)] for i in idxs]
        return (results, scores.tolist()) if return_score else results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = [self.corpus_list[int(i)] for i in flat_idxs]
            batch_results = [
                batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))
            ]

            results.extend(batch_results)
            scores.extend(batch_scores)

        return (results, scores) if return_score else results


# ---------------------------------------------------------------------------
# Page Access Actor (Ray)
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1, max_concurrency=50)
class PageAccessActor:
    """Ray actor that serves page content by URL from a local dump."""

    def __init__(self, pages_path: str | None):
        self.pages: dict[str, dict] = {}
        if pages_path is None or not os.path.exists(pages_path):
            return
        with open(pages_path, "r") as f:
            for line in f:
                page = json.loads(line)
                url = page.get("url", "")
                if url:
                    self.pages[self._normalize_url(url)] = page
                    self.pages[url] = page

    @staticmethod
    def _normalize_url(url: str) -> str:
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        if "/wiki/" in url:
            url = url.replace("/wiki/", "/w/index.php?title=")
        if "_" in url:
            url = url.replace("_", "%20")
        return url

    def access(self, urls: list[str]) -> list[dict]:
        """Batch access pages by URL."""
        resp = []
        for url in urls:
            page = self.pages.get(url)
            if page is None:
                normalized = self._normalize_url(url)
                page = self.pages.get(normalized)
            if page is None:
                resp.append({"url": url, "contents": "", "error": "Page not found"})
            else:
                if "contents" not in page and "page" in page:
                    page = {**page, "contents": page["page"]}
                resp.append(page)
        return resp
