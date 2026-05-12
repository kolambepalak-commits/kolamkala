"""
main.py — KolamKala FastAPI application entry point.

Responsibilities:
  - Initialize the FastAPI app with title and description.
  - Enable CORS so the frontend can communicate with the backend.
  - Register the generator and analyzer routers.
  - Serve the static frontend files (HTML, CSS, JS) from the same server.
  - Run Uvicorn when executed directly.

Serving order (important):
  1. CORS middleware
  2. API routers (/generate, /analyze)  ← registered first, highest priority
  3. StaticFiles mounted at "/"          ← catch-all, lowest priority
     index.html is served at "/" automatically (html=True).
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .routes.generator import router as generator_router
from .routes.analyzer import router as analyzer_router

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

class NoCacheMiddleware(BaseHTTPMiddleware):
    """Prevent browsers and CDN layers from caching any response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app = FastAPI(
    title="KolamKala API",
    description=(
        "Backend API for KolamKala — a celebration of traditional Indian Kolam art. "
        "Provides Kolam pattern generation and OpenCV-based image analysis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware — allow all origins so the frontend can connect freely
# ---------------------------------------------------------------------------

app.add_middleware(NoCacheMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Open for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routers — must be registered BEFORE the StaticFiles mount
# ---------------------------------------------------------------------------

app.include_router(generator_router, tags=["Generator"])
app.include_router(analyzer_router, tags=["Analyzer"])

# ---------------------------------------------------------------------------
# Health check — moved to /health so GET "/" is free for index.html
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    """Confirm the API server is running and list available endpoints."""
    return {
        "status": "ok",
        "service": "KolamKala Python Backend",
        "endpoints": {
            "generate": "POST /generate",
            "analyze":  "POST /analyze",
            "docs":     "GET  /docs",
        },
    }

# ---------------------------------------------------------------------------
# Static frontend — mount LAST so API routes take priority.
#
# html=True:  serves index.html for any directory request (e.g. "/").
# directory:  path relative to the workspace root where uvicorn is launched
#             (/home/runner/workspace), pointing at the shared frontend folder.
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(
    os.path.dirname(__file__),       # backend/
    "..",                            # workspace root
    "artifacts", "api-server", "frontend"
)

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)

# ---------------------------------------------------------------------------
# Entry point (run directly: python -m backend.main)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
