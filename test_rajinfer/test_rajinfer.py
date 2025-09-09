import subprocess
import pandas as pd
import re

# === Settings ===
CSV_PATH = "/home/dilreet/Desktop/Rajinfer/test_rajinfer/inference_prompts.csv"
OUTPUT_PATH = "/home/dilreet/Desktop/Rajinfer/test_rajinfer/inference_prompts.csv"
BINARY = "./build/main"
MODEL = "model_fp32_rmsnorm.yalm"

def run_inference(prompt: str, max_tokens=512, temp=0.7) -> float:
    """Runs C++ inference binary and returns throughput (tok/s)."""
    cmd = [
        BINARY,
        MODEL,
        "-i", prompt,
        "-m", "completion",
        "-n", str(max_tokens),
        "-d", "cuda",
        "-t", str(temp)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        # Search for throughput: 108.83tok/s
        match = re.search(r"throughput:\s*([\d.]+)tok/s", output)
        if match:
            return float(match.group(1))
        else:
            print("⚠️ No throughput found in output")
            return None
    except subprocess.CalledProcessError as e:
        print("❌ Error running inference:", e.stderr)
        return None

def main():
    df = pd.read_csv(CSV_PATH)

    rajinfer_speeds = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        print(f"Running Rajinfer on prompt {idx} (len={len(prompt)})...")
        speed = run_inference(prompt)
        rajinfer_speeds.append(speed)

    df["rajinfer"] = rajinfer_speeds
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Updated CSV saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()