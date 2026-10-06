"""Vercel entrypoint - FastAPI app."""
from api.vet_api import app

# Vercel expects the FastAPI app to be available as `app`
# This file is the entrypoint defined in vercel.json