import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Settings ===
CSV_PATH = "/home/dilreet/Desktop/Rajinfer/test_rajinfer/inference_prompts.csv"
OUTPUT_PNG = "/home/dilreet/Desktop/Rajinfer/test_rajinfer/inference_speeds.png"

# === Load data ===
df = pd.read_csv(CSV_PATH)

# === Line graph ===
x = np.arange(len(df))  # prompt indices (0..49)

plt.figure(figsize=(14, 6))

plt.plot(x, df["huggingface_speed"], label="HuggingFace", marker="o")
plt.plot(x, df["sglang_speed"], label="SGLang", marker="s")
plt.plot(x, df["vllm_speed"], label="vLLM", marker="^")
plt.plot(x, df["rajinfer"], label="RajInfer", marker="d")

plt.xlabel("Prompt Index (0–49)")
plt.ylabel("Tokens per Second")
plt.title("Inference Speed Comparison Across 50 Prompts")
plt.xticks(x, rotation=90, fontsize=8)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"Saved plot to {OUTPUT_PNG}")