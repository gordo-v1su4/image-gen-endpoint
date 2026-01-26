"""Main FastAPI application entry point."""

import re
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import health, create, edit


def get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup: Initialize models if needed
    print("🚀 Starting ImageGen Endpoint...")
    yield
    # Shutdown: Cleanup
    print("👋 Shutting down ImageGen Endpoint...")


VERSION = get_version()

app = FastAPI(
    title="ImageGen Endpoint",
    description="Production-ready image generation and editing API",
    version=VERSION,
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(create.router, prefix="/v1/images", tags=["Image Generation"])
app.include_router(edit.router, prefix="/v1/images", tags=["Image Editing"])

# Mount static files for test UI
app.mount("/test-ui", StaticFiles(directory="test-ui", html=True), name="test-ui")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ImageGen Endpoint",
        "version": VERSION,
        "docs": "/docs",
        "test_ui": "/test-ui/",
        "webapp": "/test-ui/webapp.html",
    }
