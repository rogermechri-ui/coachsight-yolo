"""Controle des adresses recues en entree (lot 2)."""
import socket

import pytest

import safe


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    table = {"files.example.org": "93.184.216.34", "inside.example.org": "10.0.0.5",
             "abc.supabase.co": "104.18.38.10"}

    def fake(host, port, *a, **k):
        if host not in table:
            raise OSError("unknown host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.delenv("URL_GUARD", raising=False)
    monkeypatch.delenv("ALLOWED_URL_HOSTS", raising=False)


def test_public_https_accepted():
    assert safe.url_problem("https://files.example.org/v.mp4?token=abc") is None


@pytest.mark.parametrize("url", [
    "http://files.example.org/v.mp4", "file:///etc/passwd", "/etc/passwd", "concat:/a|/b",
    "https://127.0.0.1/x", "https://169.254.169.254/latest/meta-data", "https://[::1]/x",
    "https://inside.example.org/x", "https://user:pw@files.example.org/x", "https://nowhere.example.org/x",
    "https://[::ffff:10.0.0.1]/x", 12, ""])
def test_refused(url):
    assert safe.url_problem(url)


def test_refusal_never_echoes_the_query():
    why = safe.url_problem("http://files.example.org/v.mp4?token=SECRETVALUE")
    assert "SECRETVALUE" not in why


def test_allowed_list(monkeypatch):
    monkeypatch.setenv("ALLOWED_URL_HOSTS", "supabase.co, coachsightai.com")
    assert safe.url_problem("https://abc.supabase.co/storage/v1/x") is None
    assert "allowed list" in safe.url_problem("https://files.example.org/v.mp4")
    assert safe.url_problem("https://evilsupabase.co/x")   # pas un sous-domaine


def test_check_inputs():
    ok = {"task": "hls", "video_url": "https://files.example.org/v.mp4", "callback_url": "https://files.example.org/cb",
          "segment_upload_urls": ["https://files.example.org/1", "https://files.example.org/2"], "note_url": None, "fps": 25}
    assert safe.check_inputs(ok) is None
    assert safe.check_inputs(dict(ok, video_url="/etc/passwd")).startswith("input video_url refused")
    assert safe.check_inputs(dict(ok, segment_upload_urls=["https://files.example.org/1", "https://10.0.0.1/2"])).startswith(
        "input segment_upload_urls refused")
    assert safe.check_inputs(dict(ok, callback_url="http://files.example.org/cb")).startswith("input callback_url ")


def test_check_url_raises():
    with pytest.raises(ValueError):
        safe.check_url("https://10.1.2.3/x", "video address")


def test_guard_off(monkeypatch):
    monkeypatch.setenv("URL_GUARD", "off")
    assert safe.url_problem("/etc/passwd") is None
    assert safe.check_inputs({"video_url": "/etc/passwd"}) is None
    assert safe.ffmpeg_protocols("https://x/y") == []


def test_ffmpeg_protocols():
    assert safe.ffmpeg_protocols("https://files.example.org/v.mp4") == ["-protocol_whitelist", "https,tls,tcp"]
    assert safe.ffmpeg_protocols("/tmp/v.mp4") == ["-protocol_whitelist", "file"]
