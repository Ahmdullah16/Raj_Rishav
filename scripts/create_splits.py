#!/usr/bin/env python3
"""Generate verified random 70/15/15 splits from the existing video manifest."""
import argparse
import json
from pathlib import Path
from har.data import index_directory, read_manifest, stratified_split, summary, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/video_manifest.csv")
    p.add_argument("--data-root", help="Optional: index video files on disk instead of supplied CSV")
    p.add_argument("--output", default="data/train_val_test_splits_70_15_15.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rows = index_directory(a.data_root) if a.data_root else read_manifest(a.manifest)
    split = stratified_split(rows, seed=a.seed)
    write_csv(a.output, split, ["video_path", "activity", "bytes", "split"])
    meta = summary(split, seed=a.seed)
    out = Path(a.output).with_name("dataset_summary_70_15_15.json")
    out.write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta, indent=2))
    print("Written", a.output, "and", out)


if __name__ == "__main__":
    main()
