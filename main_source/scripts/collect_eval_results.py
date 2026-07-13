import csv
import json
from pathlib import Path

import yaml

CONFIGS = [
    "configs/baseline_unet.yaml",
    "configs/baseline_unetpp.yaml",
    "configs/baseline_transunet.yaml",
    "configs/baseline_umamba.yaml",
    "configs/baseline_vmunet.yaml",
    "configs/efficient_b0_multiscale_no_se.yaml",
    "configs/efficient_b0_multiscale_boundary_no_se.yaml",
]

rows = []
for cfg_path in CONFIGS:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    out_dir = Path(cfg["paths"]["output_dir"])
    result_path = out_dir / "eval" / "test_results.json"
    if not result_path.exists():
        print(f"Skip missing: {result_path}")
        continue
    data = json.load(open(result_path, "r", encoding="utf-8"))
    row = {
        "model": cfg["project_name"],
        "config": cfg_path,
        "dice": data.get("dice"),
        "iou": data.get("iou"),
        "hd95": data.get("hd95"),
        "assd": data.get("assd"),
        "boundary_f1": data.get("boundary_f1"),
        "params": data.get("params"),
        "gflops": data.get("gflops"),
        "checkpoint_epoch": data.get("checkpoint_epoch"),
        "checkpoint_best_dice": data.get("checkpoint_best_dice"),
    }
    rows.append(row)

out_path = Path("outputs/core_baseline_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
fields = ["model", "config", "dice", "iou", "hd95", "assd", "boundary_f1", "params", "gflops", "checkpoint_epoch", "checkpoint_best_dice"]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Saved: {out_path}")
for row in rows:
    print(row)
