#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict

PLS = [96, 192, 336, 720]
DATASETS = ["ecl", "etth1", "etth2", "ettm1", "ettm2", "weather", "traffic", "solar", "exchange"]

SETTING_RE = re.compile(r"^forecast_(?P<model_id>.+?)_(?P<model>timer[^_\s]*(?:_[^_\s]+)*)_(?P<data>[^\s]+)_sl")
METRIC_RE = re.compile(r"mse:(?P<mse>[-+0-9.eE]+),\s*mae:(?P<mae>[-+0-9.eE]+)")


def infer_dataset_variant(model_id):
    parts = model_id.split("_t4_", 1)
    if len(parts) != 2:
        return None, None
    ds = parts[0]
    variant = parts[1]
    return ds, variant


def parse(path):
    rows = []
    pending = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = SETTING_RE.match(line)
            if m:
                model_id = m.group("model_id")
                ds, variant = infer_dataset_variant(model_id)
                pending = {"setting": line, "model_id": model_id, "dataset": ds, "variant": variant}
                continue
            m = METRIC_RE.match(line)
            if m and pending is not None:
                pending["mse"] = float(m.group("mse"))
                pending["mae"] = float(m.group("mae"))
                rows.append(pending)
                pending = None
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] in DATASETS and row["variant"]:
            grouped[(row["dataset"], row["variant"])].append(row)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="result_long_term_forecast.txt")
    args = parser.parse_args()

    grouped = parse(args.path)
    print("dataset,variant,n,avg_mse,avg_mae,pl96_mse,pl192_mse,pl336_mse,pl720_mse")
    for ds in DATASETS:
        variants = sorted(v for d, v in grouped.keys() if d == ds)
        for variant in variants:
            rows = grouped[(ds, variant)]
            mse = [r["mse"] for r in rows[-4:]]
            mae = [r["mae"] for r in rows[-4:]]
            padded = mse + [float("nan")] * (4 - len(mse))
            avg_mse = sum(mse) / len(mse) if mse else float("nan")
            avg_mae = sum(mae) / len(mae) if mae else float("nan")
            print(
                f"{ds},{variant},{len(mse)},{avg_mse:.6f},{avg_mae:.6f},"
                f"{padded[0]:.6f},{padded[1]:.6f},{padded[2]:.6f},{padded[3]:.6f}"
            )


if __name__ == "__main__":
    main()
