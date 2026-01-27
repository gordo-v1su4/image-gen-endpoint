"""Test script to verify model components are loaded correctly."""

import torch
from diffusers import DiffusionPipeline

def check_model_components(model_id: str):
    """Check what components are loaded for a model."""
    print(f"\n{'='*60}")
    print(f"Checking components for: {model_id}")
    print(f"{'='*60}")

    try:
        # Load with from_pretrained to see what's included
        print("\nLoading pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            variant="fp8",
            use_safetensors=True,
        )

        print("\n✅ Pipeline loaded successfully!")
        print("\nComponents found:")
        print(f"  Type: {type(pipeline).__name__}")

        # Check for common components
        components = {
            "text_encoder": "Text Encoder (CLIP/T5)",
            "text_encoder_2": "Secondary Text Encoder",
            "tokenizer": "Tokenizer",
            "tokenizer_2": "Secondary Tokenizer",
            "vae": "VAE (Variational Autoencoder)",
            "unet": "U-Net (Diffusion Model)",
            "transformer": "Transformer (Diffusion Model)",
            "scheduler": "Noise Scheduler",
        }

        for component_name, description in components.items():
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    print(f"  ✅ {description}: {type(component).__name__}")

        print(f"\n{'='*60}\n")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False

if __name__ == "__main__":
    models = [
        "Qwen/Qwen-Image-2512",
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
    ]

    print("=" * 60)
    print("MODEL COMPONENT VERIFICATION")
    print("=" * 60)
    print("\nThis script verifies that all necessary components")
    print("(text encoders, VAE, U-Net/Transformer) are loaded")
    print("when using DiffusionPipeline.from_pretrained()")

    for model in models:
        check_model_components(model)

    print("\nNOTE: This test requires:")
    print("  - HF_TOKEN environment variable set")
    print("  - Sufficient disk space for model downloads")
    print("  - Internet connection")
    print("  - GPU with CUDA (or will use CPU)")
