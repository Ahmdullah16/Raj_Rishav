import csv
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from har.data import LABELS, label_from_path, read_manifest, stratified_split


def fake_rows():
    counts = [20, 19, 19, 18, 19, 19, 18]
    return [dict(video_path=f"Root/{name.replace('_', ' ').title()}/{i}.mp4",
                 activity=name, bytes=1000 + i)
            for name, count in zip(LABELS, counts)
            for i in range(count)]


def test_all_seven_labels_remain_distinct():
    for name in LABELS:
        assert label_from_path(f"Root/{name.replace('_', ' ').title()}/clip.mp4") == name
    assert label_from_path("Dataset/Walking While Using Phone/1.mp4") != label_from_path("Dataset/Walking/1.mp4")


def test_exact_global_70_15_15_and_no_leakage():
    a = stratified_split(fake_rows(), 42)
    assert len(a) == 132
    assert Counter(r["split"] for r in a) == {"train": 92, "val": 20, "test": 20}
    assert len(set(r["video_path"] for r in a)) == 132
    for name in LABELS:
        assert {r["split"] for r in a if r["activity"] == name} == {"train", "val", "test"}
    assert a == stratified_split(fake_rows(), 42)
    assert a != stratified_split(fake_rows(), 7)


def test_rejects_duplicate_paths():
    rows = fake_rows()
    try:
        stratified_split([rows[0], rows[0]])
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Should reject duplicate video")
