"""Resume safety and local frame-cache regression tests (no GPU required)."""
import json

import numpy as np
import pytest

from har.cache import sample_cached
from har.resume import check_resume_source, write_json_atomic
from har.data import LABELS


def test_sample_cache_reuses_frames_and_invalidates_on_source_change(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video fixture identifier")
    cache = tmp_path / "cache"
    calls = []

    def sampler(path, seq_length, image_size):
        calls.append(path)
        return np.ones((seq_length, image_size, image_size, 3), dtype=np.float32)

    first = sample_cached(clip, cache_dir=cache, seq_length=2, image_size=8, sampler=sampler)
    second = sample_cached(clip, cache_dir=cache, seq_length=2, image_size=8, sampler=sampler)
    assert np.array_equal(first, second)
    assert len(calls) == 1
    clip.write_bytes(b"changed video fixture identifier")
    sample_cached(clip, cache_dir=cache, seq_length=2, image_size=8, sampler=sampler)
    assert len(calls) == 2


def test_cache_budget_and_invalid_frames(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"test")
    calls = []

    def sample(path, seq_length, image_size):
        calls.append(1)
        return np.zeros((seq_length, image_size, image_size, 3), np.float32)

    for _ in range(2):
        sample_cached(clip, cache_dir=tmp_path/"cache", limit_mb=1,
                      seq_length=20, image_size=128, sampler=sample)
    assert len(calls) == 2
    assert not list((tmp_path/"cache").glob("*.npy"))

    def invalid(path, seq_length, image_size):
        return np.zeros((1, 1), np.float32)
    with pytest.raises(ValueError, match="invalid shape"):
        sample_cached(clip, cache_dir=tmp_path/"other", sampler=invalid)


def test_resume_protects_original_model_and_class_order(tmp_path):
    source = tmp_path / "original.keras"
    source.write_bytes(b"pretend checkpoint")
    class_map = tmp_path / "classes.json"
    class_map.write_text(json.dumps(sorted(LABELS)))
    check_resume_source(source, class_map, sorted(LABELS), tmp_path/"new", 3)
    with pytest.raises(ValueError, match="overwrite"):
        check_resume_source(source, class_map, sorted(LABELS), tmp_path, 3)
    class_map.write_text(json.dumps(list(reversed(sorted(LABELS)))))
    with pytest.raises(ValueError, match="ordering"):
        check_resume_source(source, class_map, sorted(LABELS), tmp_path/"new", 3)
    with pytest.raises(ValueError, match="ordering"):
        check_resume_source(source, class_map, sorted(LABELS), tmp_path/"new", None)


def test_atomic_status_roundtrip(tmp_path):
    status = {"last_completed_epoch": 3, "fit_completed": False}
    target = tmp_path / "run" / "training_state.json"
    write_json_atomic(target, status)
    assert json.loads(target.read_text()) == status
