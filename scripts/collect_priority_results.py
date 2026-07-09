"""Collect priority experiment JSON results into one CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

CONFIGS = [
    "outputs/effb0_balanced_component_v2/eval/test_results.json",
    "outputs/effb0_balanced_component_weighted_v2/eval/test_results.json",
    "outputs/effb0_balanced_component_weighted_highres_v2/eval/test_results.json",
]

rows = []
for path_str in CONFIGS:
    p = Path(path_str)
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    row = {"result_file": str(p), **data}
    rows.append(row)

out = Path("outputs/priority_fix_summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")
else:
    print("No priority result JSON files found yet.")
