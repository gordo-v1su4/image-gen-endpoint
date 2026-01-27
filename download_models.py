#!/usr/bin/env python3
"""Download AI models using HuggingFace Hub for real inference."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import torch

# Model configurations from model_manager.py
MODELS = {
    "qwen-2512": {
        "repo_id": "Qwen/Qwen-Image-2512",
        "description": "Qwen Image-2512 - High quality (12GB VRAM)",
    },
    "qwen-nunchaku": {
        "repo_id": "QuantFunc/Nunchaku-Qwen-Image-2512",
        "description": "Quantized Qwen with 4-step LoRA (8GB VRAM)",
    },
    "flux-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "description": "FLUX.2 Klein 4B - Fast 4-step generation (8GB VRAM)",
    },
    "flux-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "description": "FLUX.2 Klein 9B - Higher quality (12GB VRAM)",
    },
}

def check_gpu():
    """Check if CUDA is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ VRAM: {total_memory:.2f} GB")
        return True
    else:
        print("✗ No GPU available - will use CPU (very slow)")
        return False

def download_model(model_name: str):
    """Download a model from HuggingFace."""
    config = MODELS.get(model_name)
    if not config:
        print(f"✗ Unknown model: {model_name}")
        return False

    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {config['repo_id']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")

    # Set cache directory
    models_path = Path(os.getenv("MODELS_PATH", "./models"))
    cache_dir = models_path / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the entire model
        print("Downloading model files...")
        local_path = snapshot_download(
            repo_id=config["repo_id"],
            cache_dir=str(cache_dir),
            resume_download=True,
        )

        print(f"✓ Model downloaded to: {local_path}")
        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if you need HuggingFace token: export HF_TOKEN=your_token")
        print("2. Verify internet connection")
        print("3. Check if model requires manual approval on HuggingFace")
        return False

def main():
    """Main download script."""
    print("="*60)
    print("AI Model Downloader for Image Generation Endpoint")
    print("="*60)

    # Check GPU
    has_gpu = check_gpu()

    if not has_gpu:
        print("\n⚠️  Warning: No GPU detected. Image generation will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Check HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✓ HuggingFace token found")
    else:
        print("⚠️  No HF_TOKEN environment variable set")
        print("   Some models may require authentication")

    # Select models to download
    print("\nAvailable models:")
    for i, (name, config) in enumerate(MODELS.items(), 1):
        print(f"  {i}. {name} - {config['description']}")

    print(f"  {len(MODELS) + 1}. All models")
    print("  0. Cancel")

    try:
        choice = input(f"\nSelect model to download (0-{len(MODELS) + 1}): ").strip()
        choice_num = int(choice)

        if choice_num == 0:
            print("Cancelled.")
            return
        elif choice_num == len(MODELS) + 1:
            # Download all
            print("\nDownloading all models...")
            for model_name in MODELS.keys():
                if not download_model(model_name):
                    print(f"\n✗ Failed to download {model_name}")
                    break
            else:
                print("\n" + "="*60)
                print("✓ All models downloaded successfully!")
                print("="*60)
        elif 1 <= choice_num <= len(MODELS):
            # Download specific model
            model_name = list(MODELS.keys())[choice_num - 1]
            if download_model(model_name):
                print("\n" + "="*60)
                print(f"✓ {model_name} downloaded successfully!")
                print("="*60)
        else:
            print("Invalid choice.")

    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Start the server: uvicorn app.main:app --reload")
    print("2. Test generation: python3 test_image_gen.py")
    print("3. Check server logs for 'Loading model...' messages")
    print("4. First generation will be slow (~30-60s for model loading)")
    print("5. Subsequent generations will be faster (~2-10s)")

if __name__ == "__main__":
    main()
