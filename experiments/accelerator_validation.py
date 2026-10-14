"""Repeat the bounded search on real hardware; preserve failures and controls."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import torch

from bench import compare_outputs, get_benchmark_config, load_kernel_module, validate_kernel_module
from optimize import optimize
from prepare import collect_environment_report, resolve_device
from reference import REFERENCE_FUNCTIONS


def run(output, device, seeds=(42, 43, 44), repeats=5, iterations=25):
    device = resolve_device(device)
    if device.startswith("mps") and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback before accelerator validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ["prepare.py", "bench.py", "optimize.py", "reference.py", "auto_kernel/timing.py"]
    sources += [f"kernels/{op}.py" for op in REFERENCE_FUNCTIONS]
    report = {
        "schema_version": 1, "device": device, "platform": platform.platform(),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "environment": collect_environment_report(), "mps_cpu_fallback_enabled": False,
        "scope": "Forward FP32 kernel checks; MPS measurements are not CUDA validation.",
        "seeds": list(seeds), "repeats": repeats, "iterations": iterations,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "runs": [],
    }
    for op in REFERENCE_FUNCTIONS:
        for seed in seeds:
            name = f"{op}-seed{seed}"
            active = output / f"{name}.py"
            active.write_bytes((root / "kernels" / f"{op}.py").read_bytes())
            row = {"kernel_type": op, "seed": seed, "directory": name}
            try:
                # Separately check accelerator outputs against a CPU reference.
                torch.manual_seed(seed)
                contract = validate_kernel_module(load_kernel_module(str(active)))
                config = get_benchmark_config(op)
                cpu = config.input_builder(contract, "cpu", config.timing_shape)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu.items()}
                with torch.inference_mode():
                    actual = contract.kernel_fn(**inputs).cpu()
                    expected = REFERENCE_FUNCTIONS[op](**cpu)
                passed, diff, detail = compare_outputs(actual, expected, config.atol, config.rtol)
                row["cpu_reference"] = {"passed": passed,
                                        "max_abs_diff": diff if math.isfinite(diff) else None,
                                        "detail": detail}
                if not passed:
                    raise ValueError("CPU-reference parity failed; search was not run")
                result = optimize(active, output / name, device, seed, repeats, iterations)
                row.update(status="completed", changed=result["changed"],
                           selected_speedup=result["selected"]["speedup"],
                           accepted=[t["name"] for t in result["trials"] if t["accepted"]])
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["runs"].append(row)
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{name}: {row['status']}", flush=True)
    report["complete"] = True
    report["passed"] = bool(report["runs"]) and all(row["status"] == "completed" for row in report["runs"])
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args(argv)
    report = run(args.output, args.device, args.seeds, args.repeats, args.iterations)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
