"""
Stress-test script for the search / access server.

Usage examples:
    # Test search endpoint with 50 concurrent requests, 200 total
    python benchmark_server.py --endpoint search --concurrency 50 --total 200

    # Test access endpoint with 100 concurrent requests, 500 total
    python benchmark_server.py --endpoint access --concurrency 100 --total 500

    # Test both endpoints simultaneously (each gets its own concurrency pool)
    python benchmark_server.py --endpoint both --concurrency 50 --total 200

    # Custom server address
    python benchmark_server.py --search-url http://109.22.128.169:8000/retrieve \
                               --access-url http://109.22.128.169:8000/access \
                               --concurrency 64 --total 500
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


DEFAULT_SEARCH_URL = "http://109.22.128.169:8000/retrieve"
DEFAULT_ACCESS_URL = "http://109.22.128.169:8000/access"

# Sample queries / urls for benchmarking
SAMPLE_QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "History of the Roman Empire.",
    "How does photosynthesis work?",
    "Latest advancements in artificial intelligence.",
    "The theory of relativity overview.",
    "Causes of World War I.",
    "Introduction to machine learning.",
    "Climate change effects on biodiversity.",
    "Overview of the Python programming language.",
]

SAMPLE_URLS = [
    "https://en.wikipedia.org/wiki/France",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Roman_Empire",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Theory_of_relativity",
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
]


@dataclass
class BenchmarkResult:
    endpoint: str
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def qps(self) -> float:
        if not self.latencies:
            return 0.0
        return self.total / sum(self.latencies)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Endpoint : {self.endpoint}")
        print(f"{'='*60}")
        print(f"Total requests : {self.total}")
        print(f"Success        : {self.success}")
        print(f"Failed         : {self.failed}")
        if self.latencies:
            print(f"QPS            : {self.qps:.2f}")
            print(f"Avg latency    : {statistics.mean(self.latencies)*1000:.2f} ms")
            print(f"Min latency    : {min(self.latencies)*1000:.2f} ms")
            print(f"Max latency    : {max(self.latencies)*1000:.2f} ms")
            sorted_lat = sorted(self.latencies)
            for p in [50, 90, 95, 99]:
                idx = int(len(sorted_lat) * p / 100) - 1
                idx = max(0, idx)
                print(f"P{p} latency     : {sorted_lat[idx]*1000:.2f} ms")
        if self.errors:
            print(f"\nFirst 5 errors:")
            for e in self.errors[:5]:
                print(f"  - {e}")
        print(f"{'='*60}\n")


async def bench_search(
    session: aiohttp.ClientSession,
    url: str,
    concurrency: int,
    total: int,
    timeout: int,
) -> BenchmarkResult:
    result = BenchmarkResult(endpoint=f"POST {url}")
    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int):
        query = SAMPLE_QUERIES[idx % len(SAMPLE_QUERIES)]
        payload = {"queries": [query], "topk": 5, "return_scores": False}
        start = time.perf_counter()
        try:
            async with sem:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    await resp.json()
            latency = time.perf_counter() - start
            result.latencies.append(latency)
            result.success += 1
        except Exception as e:
            latency = time.perf_counter() - start
            result.latencies.append(latency)
            result.failed += 1
            result.errors.append(str(e))

    tasks = [asyncio.create_task(_one(i)) for i in range(total)]
    await asyncio.gather(*tasks, return_exceptions=True)
    result.total = total
    return result


async def bench_access(
    session: aiohttp.ClientSession,
    url: str,
    concurrency: int,
    total: int,
    timeout: int,
) -> BenchmarkResult:
    result = BenchmarkResult(endpoint=f"POST {url}")
    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int):
        target_url = SAMPLE_URLS[idx % len(SAMPLE_URLS)]
        payload = {"urls": [target_url]}
        start = time.perf_counter()
        try:
            async with sem:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    resp.raise_for_status()
                    await resp.json()
            latency = time.perf_counter() - start
            result.latencies.append(latency)
            result.success += 1
        except Exception as e:
            latency = time.perf_counter() - start
            result.latencies.append(latency)
            result.failed += 1
            result.errors.append(str(e))

    tasks = [asyncio.create_task(_one(i)) for i in range(total)]
    await asyncio.gather(*tasks, return_exceptions=True)
    result.total = total
    return result


async def main():
    parser = argparse.ArgumentParser(description="Benchmark search/access server")
    parser.add_argument(
        "--endpoint",
        choices=["search", "access", "both"],
        default="both",
        help="Which endpoint to test",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--search-url",
        default=DEFAULT_SEARCH_URL,
        help="URL for the search/retrieve endpoint",
    )
    parser.add_argument(
        "--access-url",
        default=DEFAULT_ACCESS_URL,
        help="URL for the access endpoint",
    )
    args = parser.parse_args()

    connector = aiohttp.TCPConnector(limit=500, limit_per_host=500, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.endpoint == "search":
            result = await bench_search(
                session, args.search_url, args.concurrency, args.total, args.timeout
            )
            result.print_summary()
        elif args.endpoint == "access":
            result = await bench_access(
                session, args.access_url, args.concurrency, args.total, args.timeout
            )
            result.print_summary()
        elif args.endpoint == "both":
            search_task = asyncio.create_task(
                bench_search(
                    session, args.search_url, args.concurrency, args.total, args.timeout
                )
            )
            access_task = asyncio.create_task(
                bench_access(
                    session, args.access_url, args.concurrency, args.total, args.timeout
                )
            )
            search_result, access_result = await asyncio.gather(search_task, access_task)
            search_result.print_summary()
            access_result.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
