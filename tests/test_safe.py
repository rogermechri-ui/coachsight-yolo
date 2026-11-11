"""Les messages d'erreur ne doivent jamais emporter une cle ou une adresse signee."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe import scrub  # noqa: E402


def test_roboflow_key_in_url_is_removed():
    msg = ("401 Client Error: Unauthorized for url: "
           "https://detect.roboflow.com/football-field-detection-f07vi/14?api_key=AbCdEf123456XYZ")
    out = scrub(msg, 200)
    assert "AbCdEf123456XYZ" not in out
    assert "detect.roboflow.com/football-field-detection-f07vi/14" in out   # l'hote reste lisible
    assert out.startswith("401 Client Error")


def test_signed_storage_url_token_is_removed():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJ2aWRlb3MvYS9iL2MubXA0In0.c2lnbmF0dXJlLXZhbHVl"
    msg = "HTTPSConnectionPool: Max retries exceeded with url: https://x.supabase.co/storage/v1/object/sign/videos/a/b/c.mp4?token=%s (Caused by ...)" % jwt
    out = scrub(msg)
    assert jwt not in out and "eyJ" not in out
    assert "/storage/v1/object/sign/videos/a/b/c.mp4" in out
    assert "(Caused by ...)" in out


def test_truncated_url_and_bare_token_are_removed():
    # ffmpeg ne garde que la fin de son message : l'adresse peut etre coupee avant « https:// »
    out = scrub("...object/sign/videos/a.mp4?token=abc.def.ghi&download=1: Server returned 403 Forbidden")
    assert "abc.def.ghi" not in out and "403 Forbidden" in out
    assert "eyJ" not in scrub("bad header eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk")


def test_env_secret_values_are_masked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSecretValue123")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "rfSecretValue456")
    out = scrub("failed with AIzaSecretValue123 then rfSecretValue456")
    assert "AIzaSecretValue123" not in out and "rfSecretValue456" not in out


def test_plain_messages_are_unchanged_and_limit_applies():
    msg = "proxy not needed: video is already 1920x1080 (copy width 1920)"
    assert scrub(msg) == msg
    assert scrub("No space left on device") == "No space left on device"
    assert scrub(ValueError("clip 3: bad window 10.0-5.0 s")) == "clip 3: bad window 10.0-5.0 s"
    assert len(scrub("x" * 2000, 900)) == 900
    assert scrub("see https://example.com/a/b for details") == "see https://example.com/a/b for details"
