"""
Local Search Server for ASearcher in slime.

Provides two endpoints:
- POST /retrieve   -- dense/BM25 retrieval (same API as search-r1 retrieval_server.py)
- POST /access     -- lookup full page content by URL from a local pages dump

Usage:
    python local_search_server.py \
        --index_path /path/to/faiss.index \
        --corpus_path /path/to/corpus.jsonl \
        --pages_path /path/to/pages.jsonl \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --topk 5 \
        --port 8000

Then in generate_with_asearcher.py:
    search_url = "http://127.0.0.1:8000/retrieve"
    access_url = "http://127.0.0.1:8000/access"
"""

import argparse
import json
import os
import warnings

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Corpus & model helpers
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


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


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}"
                    for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            corpus = load_corpus(self.corpus_path)
            print("Pre-loading corpus into memory ...")
            self.corpus_list = [corpus[i] for i in tqdm(range(len(corpus)))]
            del corpus
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

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


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            # co.useFloat16 = True
            co.useFloat16 = False  # using FP16 can cause instability for some models (e.g. E5), so we disable by default
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        corpus = load_corpus(self.corpus_path)
        # Pre-load into memory for fast random access under concurrent requests
        print("Pre-loading corpus into memory ...")
        self.corpus_list = [corpus[i] for i in tqdm(range(len(corpus)))]
        del corpus

        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

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
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = [self.corpus_list[int(i)] for i in flat_idxs]
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        return (results, scores) if return_score else results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


# ---------------------------------------------------------------------------
# PageAccess (for /access endpoint)
# ---------------------------------------------------------------------------

class PageAccess:
    """Load a local pages dump (jsonl) and serve full content by URL."""

    def __init__(self, pages_path: str | None):
        self.pages: dict[str, dict] = {}
        if pages_path is None or not os.path.exists(pages_path):
            return
        with open(pages_path, "r") as f:
            for line in tqdm(f, desc="Loading pages"):
                page = json.loads(line)
                # Support both "url" and normalized url keys
                url = page.get("url", "")
                if url:
                    self.pages[self._normalize_url(url)] = page
                    self.pages[url] = page  # also store raw form

    @staticmethod
    def _normalize_url(url: str) -> str:
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        if "/wiki/" in url:
            url = url.replace("/wiki/", "/w/index.php?title=")
        if "_" in url:
            url = url.replace("_", "%20")
        return url

    def access(self, url: str) -> dict | None:
        """Return page dict (with at least a 'contents' or 'page' field) or None."""
        if url in self.pages:
            return self.pages[url]
        normalized = self._normalize_url(url)
        return self.pages.get(normalized)


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

class Config:
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


class AccessRequest(BaseModel):
    urls: list[str]


app = FastAPI(title="ASearcher Local Search Server")


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Dense/BM25 retrieval endpoint.

    Input:
        {"queries": ["..."], "topk": 5, "return_scores": false}
    Output:
        {"result": [[{"title": "...", "text": "...", "contents": "..."}, ...]]}
    """
    if not request.topk:
        request.topk = config.retrieval_topk

    tmp = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
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
def access_endpoint(request: AccessRequest):
    """
    Lookup full page content by URL from the local pages dump.

    Input:
        {"urls": ["https://en.wikipedia.org/wiki/Python_(programming_language)"]}
    Output:
        {"result": [{"url": "...", "contents": "...", "title": "..."}, ...]}
    """
    resp = []
    for url in request.urls:
        page = page_access.access(url) if page_access else None
        if page is None:
            resp.append({"url": url, "contents": "", "error": "Page not found"})
        else:
            # Normalize output: ensure 'contents' key exists
            if "contents" not in page and "page" in page:
                page = {**page, "contents": page["page"]}
            resp.append(page)
    return {"result": resp}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the ASearcher local search + access server.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index or BM25 Lucene index.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus jsonl file.")
    parser.add_argument("--pages_path", type=str, default=None, help="Path to pages dump jsonl (for /access endpoint).")
    parser.add_argument("--topk", type=int, default=5, help="Default number of retrieved passages.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Retriever name (e5, bge, dpr, bm25, ...).")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path to retriever model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for FAISS.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    args = parser.parse_args()

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    print("[1/3] Loading retriever ...")
    retriever = get_retriever(config)

    print("[2/3] Loading page access ...")
    page_access = PageAccess(args.pages_path)
    if args.pages_path:
        print(f"       Loaded {len(page_access.pages)} pages from {args.pages_path}")
    else:
        print("       No pages_path provided; /access endpoint will return not-found for all URLs.")

    print(f"[3/3] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
