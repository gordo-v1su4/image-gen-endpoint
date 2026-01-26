# Test UI

Simple web interface for testing the ImageGen Endpoint API locally.

## Usage

1. Start the API server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Open `index.html` in your browser (double-click or use Live Server)

3. The UI will connect to `http://localhost:8000`

## Features

- **Generate Tab**: Create images from text prompts
- **Edit Tab**: Upload and edit images with Pillow operations
- Health status indicator
- Download generated images
- Real-time error feedback

## Note

This UI is for development/testing only and is NOT included in Docker builds.
