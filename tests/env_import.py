"""Environment import and runtime smoke tests.

Run with:
  python tests/env_import.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _skip(msg: str) -> None:
    print(f"[SKIP] {msg}")


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def check_torch() -> Any:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime env dependent
        _fail(f"Failed to import torch: {exc}")

    _ok(f"torch imported: version={torch.__version__}")
    _info(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
    return torch


def check_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False.")

    device_count = torch.cuda.device_count()
    if device_count < 1:
        _fail("No CUDA devices detected by PyTorch.")

    name = torch.cuda.get_device_name(0)
    _ok(f"CUDA available: {device_count} device(s), first device='{name}'")


def _resolve_flash_attn_func() -> Any:
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:  # pragma: no cover - runtime env dependent
        _fail(f"Failed to import flash_attn: {exc}")

    version = getattr(flash_attn, "__version__", "unknown")
    _ok(f"flash_attn imported: version={version}")

    flash_attn_func = getattr(flash_attn, "flash_attn_func", None)
    if flash_attn_func is not None:
        return flash_attn_func

    try:
        interface = importlib.import_module("flash_attn.flash_attn_interface")
        flash_attn_func = getattr(interface, "flash_attn_func", None)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        _fail(f"flash_attn installed but interface import failed: {exc}")

    if flash_attn_func is None:
        _fail("flash_attn_func not found in flash_attn package.")
    return flash_attn_func


def check_flash_attn(torch: Any) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        _skip("flash-attn check is only enforced on Linux x86_64.")
        return

    flash_attn_func = _resolve_flash_attn_func()

    q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)

    try:
        out = flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=None, causal=False
        )
    except TypeError:
        # Compatibility path for older signatures.
        out = flash_attn_func(q, k, v, 0.0, None, False)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        _fail(f"flash-attn kernel call failed: {exc}")

    if isinstance(out, tuple):
        out = out[0]

    if out.shape != q.shape:
        _fail(
            f"flash-attn output shape mismatch: expected {q.shape}, got {out.shape}"
        )
    if not torch.isfinite(out).all():
        _fail("flash-attn output contains non-finite values.")

    _ok("flash-attn CUDA kernel smoke test passed.")


def main() -> int:
    _info(f"platform={platform.platform()}")
    _info(f"python={sys.version.split()[0]}")

    try:
        if platform.system() != "Linux":
            _skip(
                "Non-Linux host detected: skip checks here. "
                "Run this script on a Linux CUDA server."
            )
            return 0

        torch = check_torch()
        check_cuda(torch)
        check_flash_attn(torch)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    _ok("Environment import checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
