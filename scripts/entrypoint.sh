#!/bin/bash
set -e

echo "=============================================="
echo "IMAGE-GEN-ENDPOINT STARTING"
echo "=============================================="

# Check if models need to be downloaded
MODEL_MARKER="/root/.cache/huggingface/.models_downloaded"

if [ ! -f "$MODEL_MARKER" ]; then
    echo ""
    echo "First run - downloading models..."
    echo "This will take a few minutes..."
    echo ""
    python /app/scripts/download_models.py
    
    if [ $? -eq 0 ]; then
        touch "$MODEL_MARKER"
        echo "Models downloaded and cached!"
    else
        echo "ERROR: Model download failed!"
        exit 1
    fi
else
    echo "Models already downloaded (cached)"
fi

echo ""
echo "Starting FastAPI server..."
echo "=============================================="
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
