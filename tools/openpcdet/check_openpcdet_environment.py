"""Audit the CUDA/PyTorch/spconv environment before OpenPCDet installation."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    problems = []
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "linux": sys.platform.startswith("linux"),
    }
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_arch_list": torch.cuda.get_arch_list()
                if torch.cuda.is_available()
                else [],
            }
        )
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        else:
            problems.append("torch.cuda.is_available() is false")
    except Exception as exc:
        report["torch_error"] = repr(exc)
        problems.append("PyTorch cannot be imported")

    report["spconv"] = _version("spconv")
    if report["spconv"] is None:
        problems.append("spconv 2.x is not installed")
    elif not str(report["spconv"]).startswith("2."):
        problems.append(f"OpenPCDet requires spconv 2.x, found {report['spconv']}")

    nvcc = shutil.which("nvcc")
    report["nvcc"] = nvcc
    if nvcc:
        result = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        )
        report["nvcc_version"] = result.stdout.strip().splitlines()[-1]
    else:
        problems.append("nvcc is missing; OpenPCDet CUDA extensions cannot build")
    if not report["linux"]:
        problems.append("OpenPCDet CUDA extensions should be built in Linux/WSL, not native Windows")
    report["problems"] = problems
    print(json.dumps(report, indent=2))
    if args.strict and problems:
        raise SystemExit("Environment is not ready: " + "; ".join(problems))


if __name__ == "__main__":
    main()
