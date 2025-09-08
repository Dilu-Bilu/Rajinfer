import argparse
import os
import safetensors
from safetensors.torch import load_file, save_file
import torch

def convert_rmsnorm_to_fp32(input_yalm, output_yalm):
    """
    Load a .yalm file, convert all RMSNorm weights to fp32, and save to a new .yalm file.
    
    Args:
        input_yalm (str): Path to the input .yalm file.
        output_yalm (str): Path to the output .yalm file.
    """
    # Load the input .yalm file
    print(f"Loading {input_yalm}...")
    tensors = load_file(input_yalm)
    metadata = safetensors.safe_open(input_yalm, framework="pt").metadata()
    
    # Extract number of layers from metadata
    if "n_layers" not in metadata:
        raise ValueError("Metadata does not contain 'n_layers'")
    n_layers = int(metadata["n_layers"])
    
    # Identify and convert RMSNorm weights to fp32
    modified_tensors = {}
    rmsnorm_keys = []
    
    # Final layer norm
    final_norm_key = "model.norm.weight"
    if final_norm_key in tensors:
        print(f"Converting {final_norm_key} to fp32...")
        modified_tensors[final_norm_key] = tensors[final_norm_key].to(torch.float32)
        rmsnorm_keys.append(final_norm_key)
    
    # Per-layer attention and MLP norms
    for l in range(n_layers):
        attn_norm_key = f"model.layers.{l}.attn.norm.weight"
        mlp_norm_key = f"model.layers.{l}.mlp.norm.weight"
        
        if attn_norm_key in tensors:
            print(f"Converting {attn_norm_key} to fp32...")
            modified_tensors[attn_norm_key] = tensors[attn_norm_key].to(torch.float32)
            rmsnorm_keys.append(attn_norm_key)
        
        if mlp_norm_key in tensors:
            print(f"Converting {mlp_norm_key} to fp32...")
            modified_tensors[mlp_norm_key] = tensors[mlp_norm_key].to(torch.float32)
            rmsnorm_keys.append(mlp_norm_key)
    
    # Copy all other tensors unchanged
    for key, tensor in tensors.items():
        if key not in rmsnorm_keys:
            modified_tensors[key] = tensor
    
    # Save the modified tensors to a new .yalm file
    print(f"Saving modified tensors to {output_yalm}...")
    save_file(modified_tensors, output_yalm, metadata)
    print(f"Done! Converted {len(rmsnorm_keys)} RMSNorm weights to fp32.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RMSNorm weights in a .yalm file to fp32.")
    parser.add_argument("input", type=str, help="Path to the input .yalm file")
    parser.add_argument("output", type=str, help="Path to the output .yalm file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        parser.error(f"Input file {args.input} does not exist")
    
    convert_rmsnorm_to_fp32(args.input, args.output)