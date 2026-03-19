import asyncio
import os
from contextlib import asynccontextmanager

import httpx
import uvloop
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config_loader import load_config
from app.router import router

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=None)
    yield
    await app.state.http_client.aclose()

def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="LLM Router API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = config
    app.include_router(router)
    
    return app

config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
config = load_config(config_path)
app = create_app(config)