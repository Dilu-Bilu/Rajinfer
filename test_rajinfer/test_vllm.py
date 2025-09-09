# benchmark_vllm_batch.py
import time
import pandas as pd
from vllm import LLM, SamplingParams

# === Settings ===
MODEL_PATH = "/home/dilreet/Desktop/Rajinfer/Llama-3.2-1B"
CSV_PATH = "test_rajinfer/inference_prompts.csv"
OUTPUT_CSV_PATH = "test_rajinfer/inference_prompts.csv"

# === Init LLM ===
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_num_seqs=8,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Benchmark all prompts ===
overall_start = time.time()

for i, row in df.iterrows():
    prompt = row["prompt"]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512
    )

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    output = outputs[0].outputs[0]
    num_output_tokens = len(output.token_ids)
    elapsed = end_time - start_time
    tokens_per_sec = num_output_tokens / elapsed if elapsed > 0 else 0.0

    print(f"[{i+1}/{len(df)}] Prompt len={len(prompt)} | "
          f"Output tokens={num_output_tokens} | {tokens_per_sec:.2f} tok/s")

    df.at[i, "vllm_speed"] = tokens_per_sec

overall_end = time.time()
total_elapsed = overall_end - overall_start

print(f"\n=== Completed {len(df)} prompts ===")
print(f"Total elapsed time: {total_elapsed:.2f} seconds")
print(f"Average per prompt: {total_elapsed/len(df):.2f} seconds")

# === Save results ===
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nUpdated CSV saved to {OUTPUT_CSV_PATH}")
