#!/usr/bin/env python3
"""Quick test for FLUX.2 [klein] 4B inference."""

import requests
import base64
from pathlib import Path
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_flux_generation():
    """Test FLUX image generation."""
    print("\n" + "="*60)
    print("Testing FLUX.2 [klein] 4B - Local Inference")
    print("="*60)

    # Check health
    print("\n1. Checking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        health = resp.json()
        print(f"   ✓ Server: {health.get('ok')}")
        print(f"   ✓ CUDA: {health.get('cuda_available')}")
        if health.get('gpu_name'):
            print(f"   ✓ GPU: {health.get('gpu_name')}")
            print(f"   ✓ VRAM: {health.get('vram_used_gb', 0):.1f}GB / {health.get('vram_total_gb', 0):.1f}GB")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        print("\n   Make sure server is running:")
        print("   .venv/bin/uvicorn flux_server:app --reload")
        return

    # Test image generation
    print("\n2. Generating test image...")
    print("   Prompt: 'a cute cat wearing sunglasses, photorealistic'")
    print("   (First generation may take 30-60s for model loading)")

    try:
        resp = requests.post(
            f"{BASE_URL}/v1/images/create",
            json={
                "prompt": "a cute cat wearing sunglasses, photorealistic",
                "width": 512,
                "height": 512,
                "steps": 4,
                "guidance": 1.0,
                "seed": 42
            },
            timeout=300
        )

        if resp.status_code == 200:
            data = resp.json()
            print("   ✓ Generation successful!")

            # Save image
            if 'data' in data and 'images' in data['data']:
                b64_data = data['data']['images'][0]['data']
                image_bytes = base64.b64decode(b64_data)

                output_dir = Path("test_outputs")
                output_dir.mkdir(exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"flux_cat_{timestamp}.png"

                output_path.write_bytes(image_bytes)
                print(f"   ✓ Saved: {output_path} ({len(image_bytes)} bytes)")
                print("\n" + "="*60)
                print("✅ SUCCESS! Real AI image generated!")
                print("="*60)
                print(f"\nOpen the image: {output_path}")
                return True
        else:
            print(f"   ✗ Failed: {resp.status_code}")
            print(f"   Response: {resp.text[:200]}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    return False

if __name__ == "__main__":
    test_flux_generation()
