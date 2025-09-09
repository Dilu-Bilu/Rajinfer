# benchmark_hf_batch.py
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Settings ===
MODEL_PATH = "/home/dilreet/Desktop/Rajinfer/Llama-3.2-1B"
CSV_PATH = "test_rajinfer/inference_prompts.csv"
OUTPUT_CSV_PATH = "test_rajinfer/inference_prompts.csv"

# === Load model/tokenizer ===
print("Loading HuggingFace model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Benchmark all prompts ===
overall_start = time.time()

for i, row in df.iterrows():
    prompt = row["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512 + inputs["input_ids"].shape[-1],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    num_output_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    elapsed = end_time - start_time
    tokens_per_sec = num_output_tokens / elapsed if elapsed > 0 else 0.0

    print(f"[{i+1}/{len(df)}] Prompt len={len(prompt)} | "
          f"Output tokens={num_output_tokens} | {tokens_per_sec:.2f} tok/s")

    df.at[i, "huggingface_speed"] = tokens_per_sec

overall_end = time.time()
total_elapsed = overall_end - overall_start

print(f"\n=== Completed {len(df)} prompts ===")
print(f"Total elapsed time: {total_elapsed:.2f} seconds")
print(f"Average per prompt: {total_elapsed/len(df):.2f} seconds")

# === Save results ===
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nUpdated CSV saved to {OUTPUT_CSV_PATH}")
