"""Tiny synthetic video ZIP fixtures: no Kaggle credentials or real footage needed."""
import zipfile
from pathlib import Path

import numpy as np
import pytest

from har.video import resolve_video_file, sample_video, validate_video


def make_fixture(tmp_path):
    cv2 = pytest.importorskip('cv2')
    true_mp4 = tmp_path / 'real.mp4'
    writer = cv2.VideoWriter(str(true_mp4), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (32, 32))
    if not writer.isOpened():
        pytest.skip('FFmpeg MP4 encoder not available')
    for i in range(10):
        writer.write(np.full((32,32,3), i*15, dtype=np.uint8))
    writer.release()
    wrapped = tmp_path / 'fake.mp4'
    with zipfile.ZipFile(wrapped, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(true_mp4, arcname='inside/real.mp4')
    return true_mp4, wrapped


def test_zip_disguised_as_mp4_gets_extracted(tmp_path):
    real, wrapped = make_fixture(tmp_path)
    assert zipfile.is_zipfile(wrapped)
    cached = resolve_video_file(wrapped, cache_dir=tmp_path/'cached')
    assert cached.is_file() and cached != wrapped
    assert cached.read_bytes() == real.read_bytes()
    assert resolve_video_file(wrapped, cache_dir=tmp_path/'cached') == cached
    info = validate_video(wrapped, cache_dir=tmp_path/'cached')
    assert info['was_zip'] is True and info['width'] == 32
    # Explicit request-owned extraction dir: no shared global upload cache.
    frames = sample_video(wrapped, seq_length=20, image_size=64,
                          cache_dir=tmp_path/'request_cache')
    assert frames.shape == (20,64,64,3)
    assert float(frames.min()) >= 0 and float(frames.max()) <= 1
    assert len(list((tmp_path/'request_cache').glob('*.mp4'))) == 1


def test_real_video_passes_through(tmp_path):
    real, _ = make_fixture(tmp_path)
    assert resolve_video_file(real) == real


def test_zip_without_video_raises(tmp_path):
    z = tmp_path/'fake.mp4'
    with zipfile.ZipFile(z, 'w') as archive:
        archive.writestr('readme.txt', 'invalid')
    with pytest.raises(ValueError, match='exactly one video'):
        resolve_video_file(z, cache_dir=tmp_path/'cache')
