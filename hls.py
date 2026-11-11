"""Copie de visionnage en morceaux (HLS) pour le site : tache `hls`.

Pourquoi : la video originale d'un match fait souvent 2 a 4 Go. Le stockage
met 8 a 40 s a servir le premier octet d'un endroit du fichier jamais lu
(mesure du 02/11 sur Argenteuil, 3,4 Go) : « Watch » reste noir. En morceaux
de 6 s (~1 Mo chacun, fichiers separes), n'importe quel moment du match
demarre en 1 a 2 s.

Entree :
  video_url        adresse lisible de la video originale (signee)
  segment_s        duree d'un morceau, defaut 6
  height           hauteur de la copie, defaut 720
  crf, preset      reglages x264 (defaut 26, veryfast)
  playlist_upload_url   adresse signee PUT pour `index.m3u8` (optionnelle)
  segment_upload_urls   liste d'adresses signees PUT, une par morceau, dans
                        l'ordre (seg00000.ts, seg00001.ts...). Le site en
                        fournit ceil(duree / segment_s) + 2. S'il en manque,
                        la tache echoue proprement avant l'envoi.
  keep_local       tests locaux : garde les fichiers et renvoie le dossier
  test_duration_s  tests : ne traite que les N premieres secondes (mesure de vitesse)
Sans adresses d'envoi : fabrication seule (mesure du temps), rien n'est envoye.

La playlist cite les morceaux par leur nom seul (seg00000.ts) : le lecteur du
site remplace chaque nom par son adresse signee au moment de la lecture.
"""
import concurrent.futures as cf
import math
import os
import shutil
import subprocess
import tempfile
import time

import requests

from export import FFMPEG, _ffprobe_size
from safe import scrub

HLS_SEGMENT_S = float(os.environ.get("HLS_SEGMENT_S", "6"))
HLS_HEIGHT = int(os.environ.get("HLS_HEIGHT", "720"))
HLS_CRF = int(os.environ.get("HLS_CRF", "26"))
HLS_PRESET = os.environ.get("HLS_PRESET", "veryfast")
HLS_FPS = int(os.environ.get("HLS_FPS", "25"))
HLS_UPLOAD_WORKERS = int(os.environ.get("HLS_UPLOAD_WORKERS", "8"))


def _put(url, path, content_type, tries=4):
    last = None
    for k in range(tries):
        try:
            with open(path, "rb") as f:
                r = requests.put(url, data=f, timeout=300, headers={"Content-Type": content_type, "x-upsert": "true"})
            if r.status_code < 300:
                return
            last = "HTTP %s %s" % (r.status_code, scrub(r.text, 200))
        except Exception as e:  # reseau : on reessaie
            last = scrub(e, 200)
        time.sleep(2 * (k + 1))
    raise RuntimeError("upload failed for %s: %s" % (os.path.basename(path), last))


def build_hls(inp):
    clock = time.time()
    src = inp["video_url"]
    seg = float(inp.get("segment_s") or HLS_SEGMENT_S)
    height = int(inp.get("height") or HLS_HEIGHT)
    crf = int(inp.get("crf") or HLS_CRF)
    preset = inp.get("preset") or HLS_PRESET
    fps = int(inp.get("fps") or HLS_FPS)
    _, src_h, duration = _ffprobe_size(src)
    height = min(height, src_h) if src_h else height
    height -= height % 2
    urls_in = inp.get("segment_upload_urls") or []
    if urls_in or inp.get("playlist_upload_url"):
        # Verification avant d'encoder (12 min perdues sinon) : assez d'adresses, toutes valides.
        span = float(inp.get("test_duration_s") or duration or 0)
        need = int(math.ceil(span / seg)) if span else len(urls_in)
        bad = [i for i, u in enumerate(urls_in[:need]) if not (isinstance(u, str) and u.startswith("http"))]
        if len(urls_in) < need or bad:
            raise RuntimeError("upload URLs not usable: %d given for ~%d segments, %d missing/invalid (first: %s)"
                               % (len(urls_in), need, len(bad), bad[:5]))
        if not str(inp.get("playlist_upload_url") or "").startswith("http"):
            raise RuntimeError("playlist_upload_url missing or invalid")
    work = tempfile.mkdtemp(prefix="hls_")
    gop = int(round(seg * fps))
    net = (["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
           if str(src).startswith("http") else [])
    args = [FFMPEG, "-y", "-loglevel", "error", "-nostdin", *net,
            "-i", src, *(["-t", str(float(inp["test_duration_s"]))] if inp.get("test_duration_s") else []),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "scale=-2:%d,fps=%d" % (height, fps),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-force_key_frames", "expr:gte(t,n_forced*%g)" % seg,
            "-c:a", "aac", "-b:a", "96k", "-ac", "2",
            "-f", "hls", "-hls_time", "%g" % seg, "-hls_playlist_type", "vod",
            "-hls_segment_filename", os.path.join(work, "seg%05d.ts"),
            os.path.join(work, "index.m3u8")]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("ffmpeg hls failed: " + scrub((r.stderr or "")[-600:]))
    transcode_s = time.time() - t0
    segs = sorted(f for f in os.listdir(work) if f.endswith(".ts"))
    total = sum(os.path.getsize(os.path.join(work, f)) for f in segs)
    out = {"task": "hls", "segments": len(segs), "segment_s": seg, "height": height, "fps": fps,
           "duration_s": round(duration or 0, 1), "bytes": total,
           "avg_segment_kb": round(total / max(1, len(segs)) / 1024),
           "playlist": "index.m3u8", "segment_names": segs,
           "timings": {"transcode_s": round(transcode_s, 1),
                       "speed_x_realtime": round((float(inp.get("test_duration_s") or duration or 0)) / max(transcode_s, 0.1), 2)}}
    urls = inp.get("segment_upload_urls") or []
    if inp.get("playlist_upload_url") or urls:
        if len(urls) < len(segs):
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError("not enough segment upload URLs: %d for %d segments" % (len(urls), len(segs)))
        t1 = time.time()
        with cf.ThreadPoolExecutor(max_workers=HLS_UPLOAD_WORKERS) as ex:
            list(ex.map(lambda i: _put(urls[i], os.path.join(work, segs[i]), "video/mp2t"), range(len(segs))))
        if inp.get("playlist_upload_url"):  # en dernier : la playlist n'existe que si tout est envoye
            _put(inp["playlist_upload_url"], os.path.join(work, "index.m3u8"), "application/vnd.apple.mpegurl")
        out["uploaded"] = True
        out["timings"]["upload_s"] = round(time.time() - t1, 1)
    else:
        out["uploaded"] = False
    out["timings"]["total_s"] = round(time.time() - clock, 1)
    if inp.get("keep_local"):
        out["dir"] = work
    else:
        shutil.rmtree(work, ignore_errors=True)
    return out
