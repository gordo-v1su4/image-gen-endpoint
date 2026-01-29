#!/usr/bin/env python3
"""Pre-download models during Docker build."""

import os
import sys

def download_models():
    """Download Qwen-Image-2512, Qwen-Image-Edit-2511, and Lightning LoRAs."""
    print("=" * 60)
    print("Downloading models for pre-caching...")
    print("=" * 60)
    
    # Set cache directory
    hf_home = os.environ.get("HF_HOME", "/opt/models/huggingface")
    os.environ["HF_HOME"] = hf_home
    print(f"HF_HOME: {hf_home}")
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # 1. Download Qwen-Image-2512 (text-to-image)
        print("\n[1/4] Downloading Qwen-Image-2512 base model...")
        print("This is ~20GB and may take a while...")
        snapshot_download(
            repo_id="Qwen/Qwen-Image-2512",
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("✓ Qwen-Image-2512 downloaded!")
        
        # 2. Download Qwen-Image-2512 Lightning LoRA (bf16, 24GB-friendly)
        print("\n[2/4] Downloading Qwen-Image-2512 Lightning LoRA (bf16)...")
        hf_hub_download(
            repo_id="lightx2v/Qwen-Image-2512-Lightning",
            filename="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        )
        print("✓ Qwen-Image-2512 Lightning LoRA downloaded!")
        
        # 3. Download Qwen-Image-Edit-2511 (image editing)
        print("\n[3/4] Downloading Qwen-Image-Edit-2511 base model...")
        print("This is ~20GB and may take a while...")
        snapshot_download(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("✓ Qwen-Image-Edit-2511 downloaded!")
        
        # 4. Download Qwen-Image-Edit-2511 Lightning LoRA (bf16)
        print("\n[4/4] Downloading Qwen-Image-Edit-2511 Lightning LoRA (bf16)...")
        hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        )
        print("✓ Qwen-Image-Edit-2511 Lightning LoRA downloaded!")
        
        print("\n" + "=" * 60)
        print("All models downloaded successfully!")
        print("Models:")
        print("  - Qwen-Image-2512 + Lightning LoRA (text-to-image, 4-step)")
        print("  - Qwen-Image-Edit-2511 + Lightning LoRA (editing, 4-step)")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        print("Models will be downloaded on first use instead.")
        return 1

if __name__ == "__main__":
    sys.exit(download_models())
