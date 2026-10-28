"""
Convenience entrypoint allowing both:
  python -m uvicorn backend.main:app
and:
  python -m uvicorn backend.app.main:app
"""
from .app.main import app

__all__ = ["app"]
