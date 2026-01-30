#!/usr/bin/env python3
"""Pre-download Qwen-Image-2512 Lightning model.

Downloads ONLY the necessary files (~20GB total):
- transformer/*.safetensors (main model)
- text_encoder/ (text encoding)
- tokenizer/ (tokenization)
- vae/ (image encoding/decoding)
- config files
- Lightning LoRA (~850MB)
"""

import os
import sys

print("=" * 60)
print("DOWNLOADING QWEN-IMAGE-2512 LIGHTNING")
print("=" * 60)

HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
print(f"HF_HOME: {HF_HOME}")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    print("huggingface_hub imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

BASE_REPO = "Qwen/Qwen-Image-2512"
LORA_REPO = "lightx2v/Qwen-Image-2512-Lightning"
LORA_FILE = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"

# Only download necessary files (skip fp16/fp32 duplicates)
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer/*",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "text_encoder_2/*.safetensors", 
    "text_encoder_2/*.json",
    "text_encoder_3/*.safetensors",
    "text_encoder_3/*.json",
    "transformer/*.safetensors",
    "transformer/*.json",
    "vae/*.safetensors",
    "vae/*.json",
]

print()
print(f"[1/2] Downloading: {BASE_REPO}")
print(f"      Patterns: {len(ALLOW_PATTERNS)} file types")
print("-" * 40)

try:
    path = snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print()
print(f"[2/2] Downloading LoRA: {LORA_REPO}/{LORA_FILE}")
print("-" * 40)

try:
    lora_path = hf_hub_download(
        repo_id=LORA_REPO,
        filename=LORA_FILE,
    )
    print(f"LoRA downloaded: {lora_path}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)
