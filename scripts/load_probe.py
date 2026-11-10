"""Small parallel load probe: sends N questions at once and prints latencies and status counts.

Usage:
    uv run python scripts/load_probe.py --url http://localhost:8000 --concurrency 20

Each request is a new conversation (no session_key). Use it to see the limiter at work: with
MAX_CONCURRENT_REQUESTS below the concurrency, some requests wait for a model slot and the
ones that wait longer than CONCURRENCY_WAIT_SECONDS come back as 503 SERVER_BUSY.
"""

import argparse
import asyncio
import statistics
import time
from collections import Counter

import httpx

DEFAULT_QUESTION = "How often should I descale the KV-200 coffee maker?"


async def one(client: httpx.AsyncClient, question: str) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        response = await client.post("/api/v1/ask", json={"question": question})
        outcome = str(response.status_code)
        if response.status_code >= 400:
            outcome += " " + str(response.json().get("code", ""))
    except httpx.HTTPError as exc:
        outcome = type(exc).__name__
    return outcome.strip(), time.perf_counter() - started


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the API")
    parser.add_argument("--concurrency", type=int, default=20, help="parallel requests")
    parser.add_argument("--timeout", type=float, default=120.0, help="client timeout in seconds")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(one(client, args.question) for _ in range(args.concurrency))
        )
        total = time.perf_counter() - started

    latencies = [seconds for _, seconds in results]
    print(f"concurrency={args.concurrency} total_s={total:.2f}")
    for outcome, count in sorted(Counter(o for o, _ in results).items()):
        print(f"status {outcome}: {count}")
    print(
        f"latency_s min={min(latencies):.2f} median={statistics.median(latencies):.2f} "
        f"p95={percentile(latencies, 0.95):.2f} max={max(latencies):.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
