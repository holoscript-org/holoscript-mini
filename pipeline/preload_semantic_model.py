"""
Preload the semantic parser to avoid cold-start latency in pipeline_runner.

Usage:
    python -m pipeline.preload_semantic_model
"""
from __future__ import annotations

from pipeline.semantic_parser import get_parser


def main() -> None:
    print("[startup] Preloading semantic parser (first run downloads model ~80MB)...")
    get_parser()
    print("[startup] Preload complete.")


if __name__ == "__main__":
    main()
