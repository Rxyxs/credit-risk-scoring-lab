"""Arranca el servicio de inferencia con uvicorn.

    python run_server.py
    python run_server.py --host 0.0.0.0 --port 9000
    python run_server.py --reload
"""

from __future__ import annotations

import argparse
import sys

import uvicorn


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Recarga el servidor ante cambios de codigo (solo desarrollo).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    uvicorn.run("src.inference_api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
