"""Bounded local cache of preprocessed HAR tensors, never copies source videos."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


def sample_cached(path: str | Path, *, cache_dir: str | Path | None = None,
                  limit_mb: int = 256, seq_length: int = 20, image_size: int = 64,
                  sampler: Callable | None = None) -> np.ndarray:
    """Reuse a validated .npy sequence; fall back to decoding if cache is full.

    Identity covers the original path, size, mtime, and preprocessing version.
    132 clips at 20x64x64x3 float32 consume about 130 MB; cache stays local.
    """
    if sampler is None:
        from .video import sample_video
        sampler = sample_video
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if cache_dir is None:
        return sampler(source, seq_length=seq_length, image_size=image_size)
    if limit_mb <= 0:
        raise ValueError("Cache limit must be positive")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    identity = json.dumps(("har-sampler-v1", str(source.resolve()), stat.st_size,
                           stat.st_mtime_ns, seq_length, image_size))
    name = hashlib.sha256(identity.encode()).hexdigest()[:32] + ".npy"
    dest = cache / name
    expected_shape = (seq_length, image_size, image_size, 3)

    if dest.is_file():
        try:
            value = np.load(dest, allow_pickle=False)
            if (value.shape == expected_shape and value.dtype == np.float32
                    and np.isfinite(value).all() and value.min() >= 0
                    and value.max() <= 1):
                return value
        except (OSError, ValueError, EOFError):
            pass
        dest.unlink(missing_ok=True)

    array = np.asarray(sampler(source, seq_length=seq_length, image_size=image_size),
                       dtype=np.float32)
    if array.shape != expected_shape or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"Preprocessed video has invalid shape/range: {source.name}")
    budget = limit_mb * 1024 * 1024
    used = sum(p.stat().st_size for p in cache.glob("*.npy") if p.is_file())
    if used + array.nbytes + 256 > budget:
        return array
    with tempfile.NamedTemporaryFile(prefix="har_", suffix=".npy.tmp", dir=cache,
                                      delete=False) as handle:
        temp = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
    try:
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)
    return array
