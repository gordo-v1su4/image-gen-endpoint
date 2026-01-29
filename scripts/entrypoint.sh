#!/bin/bash
set -e

echo "🚀 Starting Image Generation Endpoint..."
echo "📦 Models will be downloaded from HuggingFace on first use"
echo "🔧 Using diffusers with CUDA acceleration"

# Start the application
echo "🌐 Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
