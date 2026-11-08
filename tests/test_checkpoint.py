"""Checkpoint recovery regression tests: no network and no TensorFlow needed."""
import hashlib
import json

import pytest

from har import checkpoint


def test_original_training_class_order():
    assert len(checkpoint.CLASS_ORDER) == 7
    assert checkpoint.CLASS_ORDER == (
        "clapping", "meet_and_split", "sitting", "standing_still", "walking",
        "walking_while_reading_book", "walking_while_using_phone",
    )


def test_download_then_reuse_verified_artifact(tmp_path, monkeypatch):
    payload = b"tiny verified fixture; not a production model"
    monkeypatch.setattr(checkpoint, "CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest())
    called = []

    def downloader(destination):
        called.append(destination)
        destination.write_bytes(payload)
        return str(destination)

    saved, classes = checkpoint.materialize_checkpoint(tmp_path, downloader)
    assert saved.read_bytes() == payload
    assert json.loads(classes.read_text()) == list(checkpoint.CLASS_ORDER)
    assert checkpoint.verify_checkpoint(saved)
    checkpoint.materialize_checkpoint(tmp_path, lambda _: pytest.fail("unexpected redownload"))
    assert len(called) == 1


def test_reject_mismatched_checkpoint_without_overwriting_previous(tmp_path, monkeypatch):
    good = b"good model checkpoint"
    monkeypatch.setattr(checkpoint, "CHECKPOINT_BYTES", len(good))
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SHA256", hashlib.sha256(good).hexdigest())
    dest = tmp_path / "best.keras"
    dest.write_bytes(b"old unrelated file")
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint.materialize_checkpoint(
            tmp_path,
            lambda partial: partial.write_bytes(b"x" * len(good)) or str(partial),
        )
    assert dest.read_bytes() == b"old unrelated file"


def test_reject_wrong_labels_for_verified_weights(tmp_path, monkeypatch):
    data = b"valid fixture"
    monkeypatch.setattr(checkpoint, "CHECKPOINT_BYTES", len(data))
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest())
    (tmp_path / "best.keras").write_bytes(data)
    (tmp_path / "classes.json").write_text('["walking","clapping"]')
    with pytest.raises(ValueError, match="class mapping"):
        checkpoint.materialize_checkpoint(tmp_path)
