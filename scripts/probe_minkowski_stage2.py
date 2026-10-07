"""Check the sparse 3D operations required before implementing Stage II.

Run this in the exact Python environment and on the GPU intended for training.
The JSON output distinguishes missing dependencies from failing operations.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch


def _exercise(me, device: str, dtype: torch.dtype | None = None) -> dict:
    coordinates = torch.tensor([
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
        [0, 0, 1, 0], [0, 2, 2, 1], [0, 4, 3, 2],
    ], dtype=torch.int32)
    features = torch.randn(len(coordinates), 4, device=device, requires_grad=True)
    stem = me.MinkowskiConvolution(4, 8, kernel_size=3, stride=1, dimension=3).to(device)
    down = me.MinkowskiConvolution(8, 8, kernel_size=2, stride=2, dimension=3).to(device)
    up = me.MinkowskiConvolutionTranspose(8, 8, kernel_size=2, stride=2, dimension=3).to(device)
    grow = me.MinkowskiGenerativeConvolutionTranspose(
        8, 8, kernel_size=2, stride=2, dimension=3).to(device)
    context = (torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None
               else torch.autocast(device_type=device, enabled=False))
    with context:
        sparse = me.SparseTensor(features=features, coordinates=coordinates)
        encoded = stem(sparse)
        coarse = down(encoded)
        decoded = up(coarse)
        expanded = grow(coarse)
        loss = encoded.F.square().mean() + coarse.F.square().mean()
        loss = loss + decoded.F.square().mean() + expanded.F.square().mean()
    loss.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise RuntimeError("Sparse backward pass produced no finite input gradients")
    return {"device": device, "autocast_dtype": str(dtype) if dtype else "none",
            "input_sites": len(sparse.C), "strided_sites": len(coarse.C),
            "transpose_sites": len(decoded.C), "generative_sites": len(expanded.C),
            "input_gradient_norm": float(features.grad.norm()),
            "finite_loss": bool(torch.isfinite(loss))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "checks": {}, "errors": {}}
    try:
        import MinkowskiEngine as me
        report["minkowski_engine"] = getattr(me, "__version__", "unknown")
    except Exception as error:
        report["minkowski_engine"] = None
        report["errors"]["import"] = repr(error)
        me = None
    if me is not None:
        for name, device, dtype in (("cpu_fp32", "cpu", None),
                                    ("cuda_fp32", "cuda", None),
                                    ("cuda_fp16", "cuda", torch.float16),
                                    ("cuda_bf16", "cuda", torch.bfloat16)):
            if device == "cuda" and not torch.cuda.is_available():
                report["errors"][name] = "CUDA unavailable"
                continue
            try:
                report["checks"][name] = _exercise(me, device, dtype)
            except Exception as error:
                report["errors"][name] = repr(error)
    report["core_ready"] = all(key in report["checks"]
                               for key in ("cpu_fp32", "cuda_fp32"))
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["core_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
