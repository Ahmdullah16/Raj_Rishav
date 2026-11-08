"""Robust video decoding for Kaggle HAR assets stored on Google Drive.

Some KaggleHub per-file downloads arrived as ZIP *archives* saved with an .mp4
extension. They are not directly decodable by OpenCV. This module detects the
actual file signature, safely extracts an enclosed MP4 to Colab's local cache,
and then samples RGB frames without modifying or duplicating the Drive data.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}
MAX_EXTRACTED_VIDEO_BYTES = 1_000_000_000  # safety cap per clip, not whole dataset


def resolve_video_file(path: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Return real video bytes, unwrapping ZIP-disguised MP4s when necessary.

    We stream exactly one video member from each ZIP archive to a deterministic
    *local* temp path (not Drive). Safe from path traversal because the archive
    member path is never used as the extraction destination. The original Drive
    file remains untouched. Files already stored as real videos pass through.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Video not found: {source}')
    if not zipfile.is_zipfile(source):
        return source

    cache = Path(cache_dir or os.environ.get('HAR_VIDEO_CACHE',
                       Path(tempfile.gettempdir()) / 'har_unpacked_videos'))
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        candidates = [item for item in archive.infolist()
                      if not item.is_dir() and Path(item.filename).suffix.lower() in VIDEO_SUFFIXES]
        if len(candidates) != 1:
            raise ValueError(
                f'{source.name} is a ZIP archive with {len(candidates)} video members; '
                'expected exactly one video, cannot select safely.')
        member = candidates[0]
        if not (0 < member.file_size <= MAX_EXTRACTED_VIDEO_BYTES):
            raise ValueError(f'Invalid/unreasonably large video inside {source}: '
                             f'{member.file_size} bytes')
        if member.flag_bits & 0x1:
            raise ValueError(f'Encrypted ZIP video not supported: {source}')
        stat = source.stat()
        identity = f'{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{member.CRC}'
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        output = cache / f'{digest}{Path(member.filename).suffix.lower()}'
        if output.is_file() and output.stat().st_size == member.file_size:
            return output
        partial = output.with_suffix(output.suffix + '.part')
        try:
            with archive.open(member, 'r') as incoming, partial.open('wb') as outgoing:
                # zipfile's reader verifies CRC as it reaches EOF.
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise ValueError(f'Extracted file size mismatch: {source.name}')
            partial.replace(output)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return output


def validate_video(path: str | Path, cache_dir: str | Path | None = None) -> dict:
    """Verify readable first frame and return helpful diagnostics."""
    import cv2
    extracted = resolve_video_file(path, cache_dir=cache_dir)
    cap = cv2.VideoCapture(str(extracted))
    try:
        if not cap.isOpened():
            raise ValueError(f'OpenCV cannot decode: {extracted} (original {path})')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError(f'Cannot read first frame from: {extracted}')
        return {'frame_count_reported': count,
                'height': int(frame.shape[0]), 'width': int(frame.shape[1]),
                'was_zip': str(Path(path)) != str(extracted)}
    finally:
        cap.release()


def sample_video(path: str | Path, seq_length: int = 20, image_size: int = 64) -> np.ndarray:
    """Return (T,H,W,3) normalized RGB frames, repeating if source is short."""
    import cv2
    extracted = resolve_video_file(path)
    cap = cv2.VideoCapture(str(extracted))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {extracted} (source: {path})')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            # Some decoders don't expose frame_count; try sequential fallback.
            frames = []
            while len(frames) < seq_length:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                raise ValueError(f'Empty or undecodable video: {path}')
            while len(frames) < seq_length:
                frames.append(frames[-1])
        else:
            indices = np.linspace(0, n - 1, seq_length).round().astype(int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f'Failed to read frame {idx} from {path}')
                frames.append(frame)
        frames = [cv2.cvtColor(cv2.resize(frame, (image_size, image_size)), cv2.COLOR_BGR2RGB)
                  for frame in frames]
        return np.asarray(frames, dtype=np.float32) / np.float32(255.0)
    finally:
        cap.release()
