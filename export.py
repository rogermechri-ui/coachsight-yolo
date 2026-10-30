"""Export video de clips (tache `export_clips`), sans dependance au GPU ni a
YOLO : utilise seulement ffmpeg. Partage entre le moteur complet (handler.py)
et le petit moteur d'export sans carte graphique (handler_export.py).

Entree : video_url, clips = [{"start_s", "end_s", "title"?}], title_cards
(carton avant chaque clip, defaut non), height (defaut 720), upload_url
(adresse signee PUT). Les clips sont mis bout a bout dans l'ordre donne.
"""
import json
import os
import tempfile
import time

import requests

FFMPEG = os.environ.get("FFMPEG_BIN") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg")

EXPORT_HEIGHT = int(os.environ.get("EXPORT_HEIGHT", "720"))
EXPORT_FPS = int(os.environ.get("EXPORT_FPS", "25"))
EXPORT_CRF = int(os.environ.get("EXPORT_CRF", "23"))
EXPORT_TITLE_S = float(os.environ.get("EXPORT_TITLE_S", "2"))
EXPORT_MAX_CLIPS = int(os.environ.get("EXPORT_MAX_CLIPS", "60"))
EXPORT_MAX_CLIP_S = float(os.environ.get("EXPORT_MAX_CLIP_S", "120"))
EXPORT_FONT = os.environ.get("EXPORT_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _ffprobe_size(src):
    """(largeur, hauteur, duree) par ffprobe, pour une source que OpenCV n'ouvre pas."""
    import subprocess, json as _json
    try:
        r = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height:format=duration", "-of", "json", src],
                           capture_output=True, text=True, timeout=120)
        d = _json.loads(r.stdout or "{}")
        st = (d.get("streams") or [{}])[0]
        return int(st.get("width") or 0), int(st.get("height") or 0), float((d.get("format") or {}).get("duration") or 0) or None
    except Exception:
        return 0, 0, None


def _run_ffmpeg(args, what):
    import subprocess
    r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s: %s" % (what, r.stderr.strip()[-300:]))


def _export_segment(src, start, dur, height, fps, crf, dst):
    """Un clip, decoupe a l'image pres (re-encode), sans son, hauteur `height`.
    -ss avant -i : ffmpeg ne lit que le passage utile, meme a distance (HTTP)."""
    net = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"] if src.startswith("http") else []
    vf = "scale=-2:'min(%d,ih)',fps=%d,setsar=1" % (height, fps)
    _run_ffmpeg([*net, "-ss", "%.3f" % max(0.0, start), "-i", src, "-t", "%.3f" % dur, "-map", "0:v:0",
                 "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", dst], "clip")


def _export_title(text, w, h, fps, crf, secs, dst):
    """Carton titre : fond sombre, texte blanc centre (lignes separees par \\n)."""
    txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    txt.write(text)
    txt.close()
    size = max(18, int(h / 14))
    draw = ("drawtext=fontfile=%s:textfile=%s:fontcolor=white:fontsize=%d:line_spacing=%d:"
            "x=(w-text_w)/2:y=(h-text_h)/2" % (EXPORT_FONT, txt.name, size, size // 2))
    try:
        _run_ffmpeg(["-f", "lavfi", "-i", "color=c=0x111418:s=%dx%d:r=%d:d=%.2f" % (w, h, fps, secs),
                     "-vf", draw + ",setsar=1", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                     "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", dst], "title")
    finally:
        os.remove(txt.name)


def export_clips(inp):
    """Fabrique un MP4 a partir d'un ou plusieurs clips de la video du match et
    l'envoie a l'adresse signee `upload_url`.
    Entree : video_url, clips = [{"start_s", "end_s", "title"?}], title_cards
    (carton avant chaque clip, defaut non), height (defaut 720), upload_url.
    Les clips sont mis bout a bout dans l'ordre donne."""
    timings = {}
    job_clock = time.time()
    src = inp["video_url"]
    clips = inp.get("clips") or []
    if not clips:
        raise RuntimeError("no clip to export")
    if len(clips) > EXPORT_MAX_CLIPS:
        raise RuntimeError("too many clips: %d (max %d)" % (len(clips), EXPORT_MAX_CLIPS))
    height = int(inp.get("height", EXPORT_HEIGHT))
    fps = int(inp.get("fps", EXPORT_FPS))
    crf = int(inp.get("crf", EXPORT_CRF))
    titles = bool(inp.get("title_cards"))
    work = tempfile.mkdtemp(prefix="export_")
    parts, done = [], []
    try:
        clock = time.time()
        for i, c in enumerate(clips):
            start, end = float(c["start_s"]), float(c["end_s"])
            dur = end - start
            if dur <= 0 or dur > EXPORT_MAX_CLIP_S:
                raise RuntimeError("clip %d: bad window %.1f-%.1f s" % (i + 1, start, end))
            seg = os.path.join(work, "c%03d.mp4" % i)
            _export_segment(src, start, dur, height, fps, crf, seg)
            if titles:
                w, h, _ = _ffprobe_size(seg)
                card = os.path.join(work, "t%03d.mp4" % i)
                _export_title(c.get("title") or "%d:%02d" % (int(start // 60), int(start % 60)),
                              w, h, fps, crf, EXPORT_TITLE_S, card)
                parts.append(card)
            parts.append(seg)
            done.append({"start_s": start, "end_s": end})
        timings["cut_s"] = round(time.time() - clock, 1)
        out = os.path.join(work, "export.mp4")
        if len(parts) == 1:
            os.replace(parts[0], out)
        else:
            lst = os.path.join(work, "list.txt")
            with open(lst, "w") as f:
                f.write("".join("file '%s'\n" % p for p in parts))
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out], "concat")
        w, h, duration = _ffprobe_size(out)
        size = os.path.getsize(out)
        clock = time.time()
        if inp.get("upload_url"):
            with open(out, "rb") as f:
                r = requests.put(inp["upload_url"], data=f, timeout=1800,
                                 headers={"Content-Type": "video/mp4", "x-upsert": "true"})
            r.raise_for_status()
            timings["upload_s"] = round(time.time() - clock, 1)
        timings["total_s"] = round(time.time() - job_clock, 1)
        result = {"task": "export_clips", "clips": len(done), "title_cards": titles, "width": w, "height": h,
                  "duration_s": round(duration, 2) if duration else None, "bytes": size, "timings": timings}
        if inp.get("keep_local"):          # tests locaux uniquement
            result["path"] = out
        return result
    finally:
        if not inp.get("keep_local"):
            import shutil
            shutil.rmtree(work, ignore_errors=True)
