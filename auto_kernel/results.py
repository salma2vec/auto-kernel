import csv
import math
from pathlib import Path
from typing import Dict, List


def read_results_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def best_speedups_by_kernel_type(path: Path) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for row in read_results_rows(path):
        if not row.get("kernel_type"):
            continue
        correctness = row.get("correctness", "")
        kept = row.get("kept", "")
        if not correctness.startswith("PASS"):
            continue
        if kept != "yes":
            continue
        try:
            speedup = float(row.get("speedup_vs_reference", "") or "1.0")
        except ValueError:
            continue
        if not math.isfinite(speedup) or speedup <= 0:
            continue
        kernel_type = row["kernel_type"]
        best[kernel_type] = max(best.get(kernel_type, 1.0), speedup)
    return best
