import argparse
import json
import os
import re
import time
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

def run_token_benchmark(
    endpoint: str,
    num_concurrent_requests: int,
    max_num_completed_requests: int,
    seq_length: int,
    test_timeout_s: int,
    results_dir: str,
    user_metadata: Dict[str, Any],
):
    headers = {'Content-Type': 'application/json'}
    data = {
        "seq_length": seq_length,
        "inputs": "Welcome to Amazon Elastic Compute Cloud,"
    }

    completed_requests = []
    num_completed_requests = 0
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)

    while time.monotonic() - start_time < test_timeout_s and len(completed_requests) < max_num_completed_requests:
        responses = []
        for _ in range(num_concurrent_requests):
            response = requests.post(endpoint, headers=headers, json=data)
            responses.append(response)

        for response in responses:
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("generated_text", "")
                num_output_tokens = len(generated_text.split())
                request_metrics = {
                    "num_input_tokens": seq_length,
                    "num_output_tokens": num_output_tokens,
                    "latency": response.elapsed.total_seconds() * 1000  # in milliseconds
                }
                completed_requests.append(request_metrics)

        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)

    pbar.close()
    end_time = time.monotonic()

    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    summary = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "endpoint": endpoint,
        "seq_length": seq_length,
        "num_concurrent_requests": num_concurrent_requests,
    }
    metadata.update(user_metadata)
    metadata["results"] = summary

    if results_dir:
        filename = f"benchmark_results_{seq_length}_{num_concurrent_requests}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        with open(results_dir / f"{summary_filename}.json", "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
            json.dump(completed_requests, f, indent=4)


def metrics_summary(metrics: List[Dict[str, Any]], start_time: int, end_time: int) -> Dict[str, Any]:
    ret = {}
    total_latency = sum(m["latency"] for m in metrics)
    total_output_tokens = sum(m["num_output_tokens"] for m in metrics)
    total_input_tokens = sum(m["num_input_tokens"] for m in metrics)

    ret["avg_latency"] = total_latency / len(metrics)
    ret["avg_throughput"] = total_output_tokens / (end_time - start_time)
    ret["avg_input_tokens"] = total_input_tokens / len(metrics)
    ret["avg_output_tokens"] = total_output_tokens / len(metrics)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a token throughput and latency benchmark.")
    parser.add_argument("--endpoint", type=str, required=True, help="The model endpoint to query.")
    parser.add_argument("--num-concurrent-requests", type=int, default=10, help="The number of concurrent requests to send (default: %(default)s)")
    parser.add_argument("--max-num-completed-requests", type=int, default=500, help="The number of requests to complete before finishing the test (default: %(default)s)")
    parser.add_argument("--seq-length", type=int, default=512, help="The sequence length to use for input (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=600, help="The amount of time to run the load test for (default: %(default)s)")
    parser.add_argument("--results-dir", type=str, default="", help="The directory to save the results to (default: %(default)s)")
    parser.add_argument("--metadata", type=str, default="", help="Additional metadata to include in the results as comma separated key=value pairs")

    args = parser.parse_args()

    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        endpoint=args.endpoint,
        num_concurrent_requests=args.num_concurrent_requests,
        max_num_completed_requests=args.max_num_completed_requests,
        seq_length=args.seq_length,
        test_timeout_s=args.timeout,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    )
