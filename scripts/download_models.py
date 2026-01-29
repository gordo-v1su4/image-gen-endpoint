#!/usr/bin/env python3
"""Pre-download models during Docker build."""

import os
import sys

def download_models():
    """Download Qwen-Image-2512 and Lightning LoRA."""
    print("=" * 60)
    print("Downloading models for pre-caching...")
    print("=" * 60)
    
    # Set cache directory
    hf_home = os.environ.get("HF_HOME", "/opt/models/huggingface")
    os.environ["HF_HOME"] = hf_home
    print(f"HF_HOME: {hf_home}")
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import torch
        
        # Download base model: Qwen-Image-2512
        print("\n[1/2] Downloading Qwen-Image-2512 base model...")
        print("This is ~20GB and may take a while...")
        snapshot_download(
            repo_id="Qwen/Qwen-Image-2512",
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("✓ Qwen-Image-2512 downloaded successfully!")
        
        # Download Lightning LoRA
        print("\n[2/2] Downloading Lightning LoRA...")
        hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename="Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
        )
        print("✓ Lightning LoRA downloaded successfully!")
        
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        print("Models will be downloaded on first use instead.")
        return 1

if __name__ == "__main__":
    sys.exit(download_models())
