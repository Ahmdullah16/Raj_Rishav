"""Verified, bounded recovery of the existing seven-class HAR Keras checkpoint.

This fetches only the published 2.8 MB checkpoint, never the video dataset.
No inference is attempted without matching labels and SHA-256 verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .data import LABELS

# Verified from the original Colab training code and saved checkpoint on 2026-11-08.
CHECKPOINT_DRIVE_ID = "1ou9A7YUm_6iwq0V_M4ul66Fz0pK23psP"
CHECKPOINT_SHA256 = "1aef90812550db788c9824699caad01379de08ab0895c3963e5f1a367f98b537"
CHECKPOINT_BYTES = 2774833
CLASS_ORDER = tuple(sorted(LABELS))
MAX_CHECKPOINT_BYTES = 12_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    return (path.is_file() and path.stat().st_size == CHECKPOINT_BYTES
            and sha256_file(path) == CHECKPOINT_SHA256)


def materialize_checkpoint(target_dir: str | Path, downloader=None) -> tuple[Path, Path]:
    """Fetch the original checkpoint once, verifying bytes before atomic install.

    Pass a downloader for tests; default is gdown for public Google Drive assets.
    No arbitrary URL or model file is trusted. Existing valid checkpoints are reused.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "best.keras"
    labels_file = target / "classes.json"

    if not verify_checkpoint(checkpoint):
        if downloader is None:
            import gdown
            downloader = lambda dest: gdown.download(id=CHECKPOINT_DRIVE_ID,
                                                        output=str(dest), quiet=True)
        with tempfile.NamedTemporaryFile(prefix="har_checkpoint_", suffix=".keras",
                                          dir=target, delete=False) as handle:
            partial = Path(handle.name)
        try:
            produced = downloader(partial)
            if not produced or not partial.exists():
                raise RuntimeError("The model download was unavailable.")
            if partial.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("Model artifact exceeds the expected size limit.")
            if not verify_checkpoint(partial):
                raise ValueError("Downloaded checkpoint failed SHA-256 or size verification.")
            os.replace(partial, checkpoint)
        finally:
            partial.unlink(missing_ok=True)

    expected = list(CLASS_ORDER)
    if labels_file.exists():
        try:
            existing = json.loads(labels_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if existing != expected:
            raise ValueError("Existing class mapping does not match the verified checkpoint.")
    else:
        labels_file.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return checkpoint, labels_file
