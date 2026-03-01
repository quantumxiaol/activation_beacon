#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List


DEFAULT_MODELS = [
    "namespace-Pt/beacon-qwen-2-7b-instruct",
]

TRAINING_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models used by this project."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all known models (inference + training examples).",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Download training example models in addition to defaults.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Custom model repo IDs, e.g. Qwen/Qwen2-7B-Instruct.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="Cache directory. Defaults to $HF_HOME if set.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token. Defaults to $HF_TOKEN.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(
            "ERROR: failed to import huggingface_hub. Run `uv sync` first.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    if args.models:
        models = _unique(args.models)
    else:
        models = list(DEFAULT_MODELS)
        if args.training or args.all:
            models.extend(TRAINING_MODELS)
        if args.all:
            # Reserved for future model additions while staying explicit.
            pass
        models = _unique(models)

    print("Models to download:")
    for model in models:
        print(f"  - {model}")

    for model in models:
        print(f"\nDownloading: {model}")
        try:
            local_path = snapshot_download(
                repo_id=model,
                cache_dir=args.cache_dir,
                token=args.token,
                resume_download=True,
            )
        except Exception as exc:
            print(f"FAILED: {model}", file=sys.stderr)
            print(f"Details: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {model}")
        print(f"Cached at: {local_path}")

    print("\nAll requested models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
