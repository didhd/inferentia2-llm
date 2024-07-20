import asyncio
import aiohttp
import time
import random
import json
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer
from datasets import load_dataset

# Llama3 tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)

def load_orca_dataset():
    print("Downloading Orca dataset...")
    dataset = load_dataset("Open-Orca/OpenOrca", data_files="1M-GPT4-Augmented.parquet", split="train")
    print("Orca dataset downloaded successfully.")
    return dataset

def generate_template_from_orca(dataset, target_tokens):
    while True:
        # 랜덤하게 데이터셋에서 항목을 선택합니다
        item = random.choice(dataset)
        text = item['question'] + " " + item['response']
        
        # 토큰화하고 목표 토큰 수에 맞게 조정합니다
        encoded = tokenizer.encode(text)
        if len(encoded) >= target_tokens:
            # 목표 토큰 수에 맞게 자릅니다
            decoded = tokenizer.decode(encoded[:target_tokens], skip_special_tokens=True)
            return decoded
        
        # 토큰 수가 부족하면 다시 선택합니다

def generate_and_save_templates(input_tokens, num_templates=5):
    print("Generating templates...")
    dataset = load_orca_dataset()
    templates = []
    for _ in range(num_templates):
        template = generate_template_from_orca(dataset, input_tokens)
        templates.append({"text": template, "tokens": len(tokenizer.encode(template))})
    
    os.makedirs('input', exist_ok=True)
    with open('input/templates.json', 'w') as f:
        json.dump(templates, f, indent=2)
    
    print("Templates generated and saved successfully.")
    return templates

def load_or_generate_templates(input_tokens):
    templates_file = 'input/templates.json'
    if os.path.exists(templates_file):
        with open(templates_file, 'r') as f:
            templates = json.load(f)
        if templates and templates[0]['tokens'] == input_tokens:
            print("Using existing templates.")
            return templates
    
    return generate_and_save_templates(input_tokens)

async def send_request(session, endpoint, data, headers):
    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    async with session.post(endpoint, json=data, headers=headers, timeout=None) as response:
        async for chunk in response.content.iter_any():
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_tokens += len(tokenizer.encode(chunk.decode()))
    end_time = time.perf_counter()
    ttft = (first_token_time - start_time) if first_token_time else None
    return response, output_tokens, ttft, start_time, end_time

async def run_token_benchmark(
    endpoint: str,
    num_concurrent_requests: int,
    max_num_completed_requests: int,
    input_tokens: int,
    max_new_tokens: int,
    test_timeout_s: int,
    results_dir: str,
    user_metadata: Dict[str, Any],
):
    headers = {'Content-Type': 'application/json'}
    
    print("Preparing benchmark data...")
    templates = load_or_generate_templates(input_tokens)
    print("Benchmark data prepared. Starting benchmark...")

    completed_requests = []
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests, desc="Benchmark Progress")

    async with aiohttp.ClientSession() as session:
        tasks = set()
        while time.monotonic() - start_time < test_timeout_s and len(completed_requests) < max_num_completed_requests:
            while len(tasks) < num_concurrent_requests and len(completed_requests) + len(tasks) < max_num_completed_requests:
                template = random.choice(templates)
                data = {
                    "inputs": template['text'],
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True
                    }
                }
                task = asyncio.create_task(send_request(session, endpoint, data, headers))
                tasks.add(task)
            
            if not tasks:
                await asyncio.sleep(0.1)
                continue

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                response, output_tokens, ttft, start_time, end_time = await task
                if response.status == 200:
                    total_time = end_time - start_time
                    
                    request_metrics = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency": total_time * 1000,  # in milliseconds
                        "ttft": ttft * 1000 if ttft else None,  # in milliseconds
                        "output_tokens_per_second": output_tokens / total_time if total_time > 0 else 0
                    }
                    completed_requests.append(request_metrics)
                    pbar.update(1)
                else:
                    print(f"Error in request: {response.status}")

    pbar.close()
    end_time = time.monotonic()

    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    summary = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "endpoint": endpoint,
        "num_concurrent_requests": num_concurrent_requests,
        "input_tokens": input_tokens,
        "max_new_tokens": max_new_tokens
    }
    metadata.update(user_metadata)
    metadata["results"] = summary

    if results_dir:
        save_results(results_dir, metadata, completed_requests)

    return metadata

def metrics_summary(metrics: List[Dict[str, Any]], start_time: float, end_time: float) -> Dict[str, Any]:
    ret = {}
    latencies = [m["latency"] for m in metrics]
    ttfts = [m["ttft"] for m in metrics if m["ttft"] is not None]
    output_tokens = [m["output_tokens"] for m in metrics]
    output_tokens_per_second = [m["output_tokens_per_second"] for m in metrics]

    ret["avg_latency"] = np.mean(latencies)
    ret["p50_latency"] = np.percentile(latencies, 50)
    ret["p90_latency"] = np.percentile(latencies, 90)
    ret["p95_latency"] = np.percentile(latencies, 95)
    ret["p99_latency"] = np.percentile(latencies, 99)

    if ttfts:
        ret["avg_ttft"] = np.mean(ttfts)
        ret["p50_ttft"] = np.percentile(ttfts, 50)
        ret["p90_ttft"] = np.percentile(ttfts, 90)
        ret["p95_ttft"] = np.percentile(ttfts, 95)
        ret["p99_ttft"] = np.percentile(ttfts, 99)

    ret["avg_output_tokens"] = np.mean(output_tokens)
    ret["avg_output_tokens_per_second"] = np.mean(output_tokens_per_second)
    ret["total_output_tokens"] = sum(output_tokens)
    ret["total_time"] = end_time - start_time
    ret["requests_per_second"] = len(metrics) / (end_time - start_time)
    ret["output_tokens_per_second"] = sum(output_tokens) / (end_time - start_time)

    return ret

def save_results(results_dir: str, metadata: Dict[str, Any], completed_requests: List[Dict[str, Any]]):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    elif not results_dir.is_dir():
        raise ValueError(f"{results_dir} is not a directory")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_filename = f"benchmark_summary_{timestamp}.json"
    individual_responses_filename = f"benchmark_individual_responses_{timestamp}.json"

    with open(results_dir / summary_filename, "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    with open(results_dir / individual_responses_filename, "w") as f:
        json.dump(completed_requests, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a token throughput and latency benchmark.")
    parser.add_argument("--endpoint", type=str, required=True, help="The model endpoint to query.")
    parser.add_argument("--num-concurrent-requests", type=int, default=8, help="The number of concurrent requests to send (default: %(default)s)")
    parser.add_argument("--max-num-completed-requests", type=int, default=100, help="The number of requests to complete before finishing the test (default: %(default)s)")
    parser.add_argument("--input-tokens", type=int, default=4096, help="Number of input tokens (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum number of new tokens to generate (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=600, help="The amount of time to run the load test for (default: %(default)s)")
    parser.add_argument("--results-dir", type=str, default="results", help="The directory to save the results to (default: %(default)s)")
    parser.add_argument("--metadata", type=str, default="", help="Additional metadata to include in the results as comma separated key=value pairs")

    args = parser.parse_args()

    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    print("Starting benchmark preparation...")
    results = asyncio.run(run_token_benchmark(
        endpoint=args.endpoint,
        num_concurrent_requests=args.num_concurrent_requests,
        max_num_completed_requests=args.max_num_completed_requests,
        input_tokens=args.input_tokens,
        max_new_tokens=args.max_new_tokens,
        test_timeout_s=args.timeout,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    ))

    print("\nBenchmark Results:")
    print(f"Concurrent requests: {args.num_concurrent_requests}")
    print(f"Input tokens: {args.input_tokens}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Avg. Latency: {results['results']['avg_latency']:.2f}ms")
    print(f"Avg. Time-to-First-Token (TTFT): {results['results']['avg_ttft']:.2f}ms")
    print(f"Output Token Throughput: {results['results']['output_tokens_per_second']:.2f} tokens/sec")
    print(f"Requests per second: {results['results']['requests_per_second']:.2f}")
    print(f"Total time: {results['results']['total_time']:.2f}s")
    print(f"Total output tokens generated: {results['results']['total_output_tokens']}")