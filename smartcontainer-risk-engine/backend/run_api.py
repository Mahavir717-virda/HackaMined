"""Tiny CLI for running the backend API server."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

APP_IMPORT = "backend.api.main:app"
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


def _ensure_project_root_on_path() -> None:
    """
    Support both:
    - python -m backend.run_api
    - python backend/run_api.py
    """
    if __package__:
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SmartContainer Risk Engine API.")
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host interface to bind (default: API_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="TCP port to bind (default: API_PORT or 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.getenv("API_LOG_LEVEL", "info"),
        help="Uvicorn log level (default: API_LOG_LEVEL or info).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_project_root_on_path()
    args = build_parser().parse_args(argv)

    uvicorn.run(
        APP_IMPORT,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
