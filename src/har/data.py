"""Dataset indexing and deterministic, per-activity 70/15/15 splitting.

No videos are duplicated, moved or modified. A manifest references each source video.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

LABELS = (
    "clapping", "meet_and_split", "sitting", "standing_still", "walking",
    "walking_while_reading_book", "walking_while_using_phone",
)
DISPLAY_NAMES = {
    "clapping": "Clapping",
    "meet_and_split": "Meet and Split",
    "sitting": "Sitting",
    "standing_still": "Standing Still",
    "walking": "Walking",
    "walking_while_reading_book": "Walking While Reading Book",
    "walking_while_using_phone": "Walking While Using Phone",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
SPLITS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)


def label_from_path(value: str) -> str:
    """Classify by the immediate parent folder (not partial filename matches)."""
    from pathlib import PurePosixPath
    parent = PurePosixPath(value.replace("\\", "/")).parent.name
    clean = "_".join(parent.lower().replace("-", " ").split())
    aliases = {"meet_and_split": "meet_and_split"}
    clean = aliases.get(clean, clean)
    if clean not in LABELS:
        raise ValueError(f"Unknown activity folder {parent!r} for {value!r}")
    return clean


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    checked = []
    seen = set()
    for row in rows:
        p = row.get("video_path", "").replace("\\", "/")
        if not p or p.startswith("/") or ".." in Path(p).parts or Path(p).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Invalid video path: {p}")
        if p in seen:
            raise ValueError(f"Duplicate source path: {p}")
        seen.add(p)
        label = label_from_path(p)
        if row.get("activity") and row["activity"] != label:
            raise ValueError(f"Label mismatch: {p}: {row['activity']} != {label}")
        size = int(row.get("bytes") or 0)
        if size <= 0:
            raise ValueError(f"Invalid byte size for {p}")
        checked.append({"video_path": p, "activity": label, "bytes": size})
    return checked


def index_directory(root: str | Path) -> list[dict]:
    """Read an extracted Kaggle folder, retaining paths relative to root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            rel = path.relative_to(root).as_posix()
            try:
                label = label_from_path(rel)
            except ValueError:
                continue  # reject unrelated video folders instead of guessing labels
            if path.stat().st_size:
                rows.append({"video_path": rel, "activity": label, "bytes": path.stat().st_size})
    if not rows:
        raise ValueError("No labeled videos found in the directory")
    return rows


def stratified_split(rows: Iterable[dict], seed: int = 42) -> list[dict]:
    """Random per-class split, close to 70/15/15 with exact global quotas.

    The fixed seed makes the video-to-split allocation reproducible. All three
    splits contain each class when class size >= 3. The final target totals
    follow largest-remainder rounding across the full dataset.
    """
    items = [dict(r) for r in rows]
    if not items:
        raise ValueError("Cannot split an empty dataset")
    seen = [r["video_path"] for r in items]
    if len(set(seen)) != len(seen):
        raise ValueError("Duplicate videos in manifest")
    grouped: dict[str, list[dict]] = {}
    for row in items:
        grouped.setdefault(row["activity"], []).append(row)
    if any(len(group) < 3 for group in grouped.values()):
        raise ValueError("Each category requires >=3 videos for three nonempty splits")
    rng = random.Random(seed)
    labels = sorted(grouped)
    for label in labels:
        grouped[label].sort(key=lambda r: r["video_path"])
        rng.shuffle(grouped[label])

    n = len(items)
    raw_totals = [n * f for f in FRACTIONS]
    totals = [int(x) for x in raw_totals]
    for i in sorted(range(3), key=lambda i: (-(raw_totals[i] - totals[i]), i))[:n - sum(totals)]:
        totals[i] += 1

    counts = {}
    for label in labels:
        m = len(grouped[label])
        ideal = [m * f for f in FRACTIONS]
        quota = [int(x) for x in ideal]
        for i in sorted(range(3), key=lambda i: (-(ideal[i] - quota[i]), i))[:m - sum(quota)]:
            quota[i] += 1
        counts[label] = quota

    current = [sum(counts[label][i] for label in labels) for i in range(3)]
    # Rebalance per-class counts to globally rounded target, minimizing error
    # versus each category's ideal fractional quotas.
    while current != totals:
        donors = [i for i in range(3) if current[i] > totals[i]]
        receivers = [i for i in range(3) if current[i] < totals[i]]
        choices = []
        for donor in donors:
            for receiver in receivers:
                for label in labels:
                    quota = counts[label]
                    if quota[donor] <= 1:
                        continue
                    ideal = [len(grouped[label]) * f for f in FRACTIONS]
                    before = sum((quota[j] - ideal[j])**2 for j in (donor, receiver))
                    after = ((quota[donor]-1 - ideal[donor])**2
                             + (quota[receiver]+1 - ideal[receiver])**2)
                    choices.append((round(after - before, 9), label, donor, receiver))
        if not choices:
            raise RuntimeError("Could not satisfy target quotas")
        _, label, donor, receiver = min(choices)
        counts[label][donor] -= 1
        counts[label][receiver] += 1
        current[donor] -= 1
        current[receiver] += 1

    result = []
    for label in labels:
        offset = 0
        for split, count in zip(SPLITS, counts[label]):
            for row in grouped[label][offset:offset + count]:
                result.append({**row, "split": split})
            offset += count
    assert len(result) == len(items)
    assert len({r["video_path"] for r in result}) == len(items)
    assert tuple(Counter(r["split"] for r in result)[s] for s in SPLITS) == tuple(totals)
    return sorted(result, key=lambda r: (r["activity"], r["video_path"]))


def write_csv(path: str | Path, rows: list[dict], columns: tuple[str, ...] | list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def summary(rows: list[dict], seed: int) -> dict:
    return {
        "source": "Kaggle: sharjeelmazhar/human-activity-recognition-video-dataset",
        "dataset_is_not_ucf101": True,
        "seed": seed,
        "video_count": len(rows),
        "total_video_bytes": sum(int(r["bytes"]) for r in rows),
        "classes": sorted(set(r["activity"] for r in rows)),
        "splits": {split: sum(r["split"] == split for r in rows) for split in SPLITS},
        "counts_by_class_and_split": {
            label: {split: sum(r["activity"] == label and r["split"] == split for r in rows)
                    for split in SPLITS}
            for label in sorted(set(r["activity"] for r in rows))
        },
        "warning": "Random video-level splits can leak actors/scenes. Use group-based splits where group IDs exist.",
        "results_note": "No model training/evaluation metrics are implied by this split.",
    }
