"""Energie sonore d'un match : tache `audio` du moteur d'export (sans GPU).

Pourquoi : les actions rapides (coup franc, corner, tir, grosse occasion) sont
souvent invisibles sur une image fixe de plan large, mais elles s'entendent :
sifflet de l'arbitre, reaction du public, ton du commentaire. On mesure le
niveau sonore seconde par seconde pour reperer des « moments candidats ».

Entree :
  video_url        adresse lisible de la video (signee) ou chemin local
  step_s           pas de mesure en secondes (defaut 1)
  start_s, test_duration_s   facultatifs : ne traiter qu'un morceau (tests)
Sortie :
  has_audio, step_s, n, duration_s,
  full     niveau global (dB RMS) par pas
  whistle  bande 2,5-4,5 kHz (sifflet)
  crowd    bande 150-1200 Hz (public, voix)
  timings  {analyse_s}
Les niveaux sont arrondis a 0,1 dB ; -120 = silence. Pas de rappel obligatoire
(mode direct depuis la console RunPod) ; avec callback_url le handler renvoie
le resultat par la meme voie que les autres taches.
"""
import os
import re
import subprocess
import tempfile
import time

from export import FFMPEG
from safe import scrub

FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def _has_audio(src):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", src], capture_output=True, text=True)
    return bool((r.stdout or "").strip())


def _read_levels(path):
    """Fichier ametadata -> liste de niveaux dB, une valeur par trame."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        m = re.search(r"lavfi\.astats\.Overall\.RMS_level=(\S+)", line)
        if m:
            v = m.group(1)
            try:
                x = float(v)
            except ValueError:
                x = -120.0
            if x != x or x < -120 or x == float("-inf"):
                x = -120.0
            out.append(round(x, 1))
    return out


def audio_energy(inp):
    t0 = time.time()
    src = inp["video_url"]
    step = float(inp.get("step_s") or 1)
    rate = 16000
    n = int(round(rate * step))
    if not _has_audio(src):
        return {"task": "audio", "has_audio": False}
    work = tempfile.mkdtemp(prefix="audio_")
    files = {k: os.path.join(work, k + ".txt") for k in ("full", "whistle", "crowd")}
    meta = "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=%s"
    graph = ("[0:a:0]aresample=%d,aformat=sample_fmts=fltp:channel_layouts=mono,asetnsamples=n=%d:p=0,asplit=3[a][b][c];"
             "[a]%s[oa];"
             "[b]highpass=f=2500,lowpass=f=4500,%s[ob];"
             "[c]highpass=f=150,lowpass=f=1200,%s[oc]") % (rate, n, meta % files["full"], meta % files["whistle"], meta % files["crowd"])
    pre = []
    if inp.get("start_s"):
        pre += ["-ss", str(float(inp["start_s"]))]
    net = (["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
           if str(src).startswith("http") else [])
    args = [FFMPEG, "-y", "-loglevel", "error", "-nostdin", *net, *pre, "-i", src,
            *(["-t", str(float(inp["test_duration_s"]))] if inp.get("test_duration_s") else []),
            "-vn", "-filter_complex", graph,
            "-map", "[oa]", "-f", "null", "-", "-map", "[ob]", "-f", "null", "-", "-map", "[oc]", "-f", "null", "-"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg audio failed: " + scrub((r.stderr or "")[-600:]))
    lv = {k: _read_levels(p) for k, p in files.items()}
    for p in files.values():
        try:
            os.remove(p)
        except OSError:
            pass
    os.rmdir(work)
    m = min(len(v) for v in lv.values())
    return {"task": "audio", "has_audio": True, "step_s": step, "start_s": float(inp.get("start_s") or 0), "n": m,
            "duration_s": round(m * step, 1),
            "full": lv["full"][:m], "whistle": lv["whistle"][:m], "crowd": lv["crowd"][:m],
            "timings": {"analyse_s": round(time.time() - t0, 1)}}
