#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ModelArgs, get_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test a locally trained Qwen3 Beacon model on long context input.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Local model path. If omitted, auto-detects the newest local Qwen3 checkpoint under data/outputs.",
    )
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/toy/infbench.json",
        help="Path to a json file with keys: context, answer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument("--chat_template", type=str, default="hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    return parser.parse_args()


def _is_qwen3_model_dir(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False

    model_type = str(config.get("model_type", "")).lower()
    architectures = config.get("architectures") or []
    arch_text = " ".join(architectures).lower()
    return "qwen3" in model_type or "qwen3" in arch_text or "qwen3" in str(model_dir).lower()


def resolve_model_path(model_name_or_path: str | None) -> Path:
    if model_name_or_path:
        candidate = Path(model_name_or_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Model path does not exist: {candidate}")
        return candidate

    env_path = os.getenv("BEACON_MODEL_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"BEACON_MODEL_PATH does not exist: {candidate}")
        return candidate

    outputs_dir = PROJECT_ROOT / "data" / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(
            f"{outputs_dir} does not exist. Pass --model_name_or_path to load your local trained model."
        )

    candidates = [p for p in outputs_dir.rglob("*") if p.is_dir() and _is_qwen3_model_dir(p)]
    if not candidates:
        raise FileNotFoundError(
            "No local Qwen3 model found under data/outputs. Pass --model_name_or_path to specify your model."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@torch.no_grad()
def main():
    args = parse_args()

    use_cpu = args.cpu or not torch.cuda.is_available()
    if use_cpu and args.attn_impl == "flash_attention_2":
        args.attn_impl = "sdpa"
        print("[Warn] CPU mode detected, attn_impl switched to sdpa.")

    model_path = resolve_model_path(args.model_name_or_path)
    print(f"[Info] Loading model from: {model_path}")

    model_args = ModelArgs(
        model_name_or_path=str(model_path),
        enable_beacon=True,
        attn_impl=args.attn_impl,
        dtype=args.dtype,
        chat_template=args.chat_template,
        cpu=use_cpu,
    )
    device = "cpu" if use_cpu else args.device
    model, tokenizer = get_model_and_tokenizer(model_args, device=device)
    model.eval()

    example_path = Path(args.example_path).expanduser()
    if not example_path.is_absolute():
        example_path = PROJECT_ROOT / example_path
    with example_path.open("r", encoding="utf-8") as f:
        example = json.load(f)

    if hasattr(model, "memory"):
        model.memory.reset()

    messages = [{"role": "user", "content": example["context"]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        top_p=1.0,
        temperature=1.0,
        max_new_tokens=args.max_new_tokens,
    )[:, inputs["input_ids"].shape[1] :]

    print("*" * 20)
    print(f"Input Length: {inputs['input_ids'].shape[1]}")
    print(f"Answers:      {example['answer']}")
    print(f"Prediction:   {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
