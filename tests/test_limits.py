"""Limites de taille (lot 3a)."""
import io

import pytest

import safe


def test_default_video_limit(monkeypatch):
    monkeypatch.delenv("MAX_VIDEO_GB", raising=False)
    assert safe.max_video_bytes() == 12 << 30


def test_video_limit_from_env(monkeypatch):
    monkeypatch.setenv("MAX_VIDEO_GB", "4")
    assert safe.max_video_bytes() == 4 << 30
    monkeypatch.setenv("MAX_VIDEO_GB", "0")
    assert safe.max_video_bytes() == 0
    monkeypatch.setenv("MAX_VIDEO_GB", "abc")
    assert safe.max_video_bytes() == 12 << 30


def test_check_size():
    assert safe.check_size(10, 100) == 10
    assert safe.check_size(None, 100) is None
    assert safe.check_size(10 ** 12, 0) == 10 ** 12      # 0 = sans limite
    with pytest.raises(ValueError, match="video too large"):
        safe.check_size(13 << 30, 12 << 30, "video")


def test_read_limited():
    assert safe.read_limited(io.BytesIO(b"x" * 10), 10) == b"x" * 10
    with pytest.raises(ValueError, match="too large"):
        safe.read_limited(io.BytesIO(b"x" * (2 << 20)), 1 << 20, "segment")
