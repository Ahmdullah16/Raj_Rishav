"""Streamlit data, checkpoint, and result helpers (no implicit network calls)."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

from .data import DISPLAY_NAMES, LABELS

REQUIRED_SPLIT_COLUMNS = {'video_path', 'activity', 'split'}
SPLIT_NAMES = ('train', 'val', 'test')
MODEL_VARIANTS = [
    {'model': 'CNN–LSTM 1', 'key': 'cnn_lstm_1', 'cnn': '1 × Conv2D(32)', 'temporal': 'LSTM(64)', 'embedding': 'Flatten', 'status': 'Requires training'},
    {'model': 'CNN–LSTM 2', 'key': 'cnn_lstm_2', 'cnn': '1 × Conv2D(32)', 'temporal': 'LSTM(64)', 'embedding': 'Global average pooling', 'status': 'Requires training'},
    {'model': 'CNN–LSTM 3', 'key': 'cnn_lstm_3', 'cnn': 'Conv2D(32/64/128)', 'temporal': 'LSTM(128)', 'embedding': 'Global average pooling', 'status': 'Requires training'},
    {'model': 'MobileNetV2–LSTM', 'key': 'mobilenetv2_lstm', 'cnn': 'Frozen ImageNet MobileNetV2', 'temporal': 'LSTM(128)', 'embedding': 'Pretrained feature pooling', 'status': 'Requires training'},
]


def load_split_table(path: str | Path) -> list[dict]:
    """Read and validate a split, rejecting duplicates and label mismatches."""
    with open(path, newline='', encoding='utf8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_SPLIT_COLUMNS <= set(reader.fieldnames):
            raise ValueError('Expected columns video_path, activity and split')
        rows = list(reader)
    if not rows:
        raise ValueError('Split file is empty')
    paths = [r['video_path'] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate video paths in split file')
    if any(r['split'] not in SPLIT_NAMES for r in rows):
        raise ValueError('Unexpected split label')
    if any(r['activity'] not in LABELS for r in rows):
        raise ValueError('Unexpected activity label')
    # Labels must match the corresponding activity folder (not prefix matches).
    from .data import label_from_path
    if any(label_from_path(r['video_path']) != r['activity'] for r in rows):
        raise ValueError('One or more video labels do not match the source folder')
    return rows


def get_dataset_summary(rows: list[dict]) -> dict:
    by_class = {label: {split: 0 for split in SPLIT_NAMES} for label in LABELS}
    split_counts = {s: 0 for s in SPLIT_NAMES}
    total = 0
    for row in rows:
        by_class[row['activity']][row['split']] += 1
        split_counts[row['split']] += 1
        total += int(row.get('bytes') or 0)
    return {'total_videos': len(rows), 'split_counts': split_counts, 'by_class': by_class,
            'total_bytes': total, 'classes': len([c for c in LABELS if sum(by_class[c].values()) > 0])}


def read_optional_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(encoding='utf8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return data


def read_json_classes(data: str | bytes) -> list[str]:
    """Only accept an explicit trained-checkpoint class ordering."""
    obj = json.loads(data)
    if isinstance(obj, dict):
        obj = obj.get('classes')
    if not isinstance(obj, list) or not obj or any(not isinstance(x, str) for x in obj):
        raise ValueError('classes.json must be a JSON list of class names or {"classes": [...]}.')
    if len(obj) != len(set(obj)):
        raise ValueError('Duplicate class labels in classes.json')
    if any(label not in LABELS for label in obj):
        raise ValueError('class names do not match the seven-class project')
    return obj


def load_classes(path: str | Path) -> list[str] | None:
    path = Path(path)
    return read_json_classes(path.read_bytes()) if path.is_file() else None


def read_optional_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf8') as fh:
        return list(csv.DictReader(fh))


def matrix_from_csv(path: str | Path, classes: list[str]):
    """Read a labeled NxN matrix; never generate synthetic metrics."""
    import pandas as pd
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p, index_col=0)
    if set(df.index.astype(str)) != set(classes) or set(df.columns.astype(str)) != set(classes):
        raise ValueError('Confusion matrix labels do not match expected checkpoint classes')
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.loc[classes, classes]
    return df.apply(pd.to_numeric, errors='raise')


def dataframe_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b''
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf8')


def size_label(value: int | float) -> str:
    return f'{value / 1_000_000_000:.3f} GB'


def display_label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace('_', ' ').title())
