#!/bin/bash
set -e

echo "🚀 Starting Image Generation Endpoint..."

# Download models if needed (on first run)
/app/scripts/download_models.sh

# Start the application
echo "🌐 Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
