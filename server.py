"""
server.py — Production entry point
Mounts FastAPI routes onto the Dash Flask server so both
run on the same port/process on Render.com
"""
from app import app, server
from api_routes import api
from fastapi.middleware.wsgi import WSGIMiddleware

# Mount Dash (WSGI) inside FastAPI (ASGI)
api.mount("/", WSGIMiddleware(server))

# Export for Gunicorn: gunicorn server:api
# (uvicorn handles ASGI for FastAPI; Dash still works via WSGIMiddleware)
