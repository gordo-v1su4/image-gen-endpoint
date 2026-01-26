"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, create, edit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup: Initialize models if needed
    print("🚀 Starting ImageGen Endpoint...")
    yield
    # Shutdown: Cleanup
    print("👋 Shutting down ImageGen Endpoint...")


app = FastAPI(
    title="ImageGen Endpoint",
    description="Production-ready image generation and editing API",
    version="0.1.0",
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ImageGen Endpoint",
        "version": "0.1.0",
        "docs": "/docs",
    }
