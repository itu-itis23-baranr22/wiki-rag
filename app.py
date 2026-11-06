"""Start the Wiki-RAG web interface:  python app.py"""
from __future__ import annotations

import argparse
import webbrowser
from threading import Timer

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Wiki-RAG web interface.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"Wiki-RAG is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        Timer(1.0, webbrowser.open, args=[url]).start()
    # Local only by design. See src/server.py before changing the host.
    uvicorn.run("src.server:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
