"""Small HTTP load smoke test for the Phase 3 prediction API."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import requests


def run(endpoint: str, requests_count: int, output: Path | None) -> str:
    """Send repeated prediction requests and return a short text report."""
    latencies: list[float] = []
    failures = 0
    payload = {
        "text": "Urgent account verification required. Click the secure link now.",
    }
    for _ in range(requests_count):
        started = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            failures += 1
        finally:
            latencies.append((time.perf_counter() - started) * 1000)

    report = "\n".join(
        [
            f"endpoint: {endpoint}",
            f"requests: {requests_count}",
            f"failures: {failures}",
            f"mean_latency_ms: {statistics.mean(latencies):.2f}",
            f"p95_latency_ms: {_quantile(latencies, 0.95):.2f}",
        ]
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
    return report


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-smoke the prediction API")
    parser.add_argument("--endpoint", required=True, help="Full /predict endpoint URL")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(run(args.endpoint, args.requests, args.output))


if __name__ == "__main__":
    main()
