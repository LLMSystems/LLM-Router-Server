import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config_loader import load_config
from app.router import router


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="LLM Router API", version="0.1.0")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)

    server_cfg = config.get("server", {})

    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8947),
        reload=False,
        log_level=server_cfg.get("uvicorn_log_level", "info")
    )


if __name__ == "__main__":
    main()
