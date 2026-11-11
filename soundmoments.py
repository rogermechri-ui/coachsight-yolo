"""Moments proposes par le son + lecture de clips : tache `sound_moments` du moteur d'export (sans GPU).

Enchainement (tout sur le moteur d'export CPU) :
  1. `audio` : niveau sonore seconde par seconde (global, sifflet, public) ;
  2. candidats : instants ou le son monte (pic a +-3 s moins mediane locale a +-40 s), limites aux
     periodes de jeu, un candidat par fenetre de `nms_s` secondes, les `n_candidates` meilleurs ;
  3. `clips_ai` : un clip de `clip_s` secondes autour de chaque candidat, lu par un modele video ;
  4. les evenements lus sont filtres (types utiles au coach, confiance minimale), fusionnes (meme type
     a moins de 4 s), puis ecartes s'ils existent deja (`existing`, meme type a +-`dedup_s`).
Resultat : des moments PROPOSES (le coach confirme dans Review) ; rien n'est ecrit par le moteur.

Entree :
  video_url          adresse lisible (signee) de la video (ou segments_json_url)
  segments_json_url  adresse signee d'un JSON {"segment_s":6,"urls":[...]} : lit la copie HLS (rapide)
  model              obligatoire (ex. "gemini-3.1-pro-preview"), aucun defaut cache
  play_windows       [[debut_s, fin_s], ...] periodes de jeu (sinon toute la video)
  n_candidates       defaut 250 ; nms_s defaut 6 ; clip_s defaut 6
  thinking_budget    defaut 128 (le modele Pro ne peut pas desactiver la reflexion)
  home_colors, away_colors   couleurs ; « home » = l'equipe du coach
  keep_types         liste de types gardes (defaut : buts, tirs, occasions, centres, corners, coups francs,
                     penalties, fautes, arrets, pertes de balle, tacles, contres, duels aeriens)
  min_confidence     0..1 (defaut 0.7)
  existing           [{"t": secondes, "type": "..."}] moments deja connus, pour ne pas les redoubler
  dedup_s            defaut 6
  cand_offset, cand_limit  tranche des candidats (ordre chronologique) pour traiter en plusieurs fois ; deadline_s (1800)
  result_upload_url  adresse signee d'envoi (PUT) : le resultat complet y est depose ; compact_output=true : la sortie RunPod n'a plus la liste
  max_clips          garde-fou de cout (defaut 300) ; workers (defaut 4)
Sortie : {task, model, n_candidates, n_clips, moments:[{t, type, team, confidence, note, source:"clip"}],
          dropped:{type, confidence, existing, merged}, usage{...}, timings}
La cle Gemini est lue dans GEMINI_API_KEY (jamais dans l'entree ni le resultat).
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import json

import audio
import clipsai
from export import FFMPEG
from safe import scrub, check_url, ffmpeg_protocols, read_limited

DEFAULT_KEEP = ["goal", "shot", "big_chance", "save", "cross", "corner", "free_kick", "penalty", "foul",
                "turnover", "tackle", "counter_attack", "aerial_duel"]
# types lus par le modele -> codes de la bibliotheque d'actions du site
CODE = {"sub": "substitution"}


MAX_SEGMENTS = int(os.environ.get("MAX_SEGMENTS", "6000"))      # 6000 x 6 s = 10 h
MAX_SEGMENT_MB = int(os.environ.get("MAX_SEGMENT_MB", "200"))


class HlsSource:
    """Copie de visionnage HLS lue segment par segment (petits fichiers, reponse rapide partout, la
    video d'origine met 8 a 40 s a repondre sur une zone jamais lue). `segments_json_url` : adresse
    signee d'un JSON {"segment_s": 6, "urls": [adresse signee de seg00000.ts, seg00001.ts, ...]}."""

    def __init__(self, segments_json_url, workers=4):
        with urllib.request.urlopen(check_url(segments_json_url, "segments list address"), timeout=60) as r:
            j = json.loads(read_limited(r, 16 << 20, "segments list"))
        self.seg_s = float(j.get("segment_s") or 6)
        self.urls = j["urls"]
        if not isinstance(self.urls, list) or len(self.urls) > MAX_SEGMENTS:
            raise ValueError("segments list refused: more than %d segments" % MAX_SEGMENTS)
        self.dir = tempfile.mkdtemp(prefix="hls_")
        self.workers = workers

    def path(self, i):
        return os.path.join(self.dir, "seg%05d.ts" % i)

    def _get(self, i):
        p = self.path(i)
        if os.path.exists(p):
            return p
        last = None
        check_url(self.urls[i], "segment address")   # refus immediat, sans nouvelle tentative
        for k in range(10):
            try:
                with urllib.request.urlopen(self.urls[i], timeout=60) as r:
                    data = read_limited(r, MAX_SEGMENT_MB << 20, "segment")
                open(p + ".part", "wb").write(data)
                os.replace(p + ".part", p)
                return p
            except ValueError:      # segment trop gros : refus immediat, sans nouvelle tentative
                raise
            except Exception as e:  # 429 (trop de demandes) : on attend de plus en plus longtemps
                last = e
                time.sleep(min(30, 2 * (k + 1)))
        raise RuntimeError("segment %d: %s" % (i, scrub(last, 120)))

    def fetch_all(self):
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            list(ex.map(self._get, range(len(self.urls))))

    def concat_all(self):
        out = os.path.join(self.dir, "all.ts")
        with open(out, "wb") as o:
            for i in range(len(self.urls)):
                with open(self._get(i), "rb") as f:
                    shutil.copyfileobj(f, o)
        return out

    def cutter(self):
        def cut(t, clip_s, out):
            start = max(0.0, t - clip_s / 2.0)
            i0 = int(start // self.seg_s)
            i1 = min(len(self.urls) - 1, int((start + clip_s) // self.seg_s))
            files = [self._get(i) for i in range(i0, i1 + 1)]
            args = [FFMPEG, "-y", "-loglevel", "error", "-nostdin",
                    *(["-protocol_whitelist", "concat,file"] if ffmpeg_protocols("") else []), "-i", "concat:" + "|".join(files),
                    "-ss", "%.2f" % (start - i0 * self.seg_s), "-t", str(clip_s), "-an",
                    "-vf", "scale=-2:480,fps=10", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-movflags", "+faststart", out]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                raise RuntimeError("ffmpeg clip failed: " + scrub((r.stderr or "")[-300:]))
        return cut

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def _median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _rise(levels, i, peak_w=3, base_w=40):
    """Montee a l'instant i : maximum a +-peak_w moins mediane a +-base_w. Seul le pic lui-meme (premier
    maximum de sa fenetre) compte, sinon un meme bruit donnerait plusieurs candidats voisins."""
    lo, hi = max(0, i - peak_w), min(len(levels), i + peak_w + 1)
    win = levels[lo:hi]
    peak = max(win)
    if lo + win.index(peak) != i:
        return float("-inf")
    base = _median(levels[max(0, i - base_w):min(len(levels), i + base_w + 1)])
    return peak - base


def candidates(levels_by_band, step_s, windows, n, nms_s):
    """Meilleurs instants (secondes) triees par montee du son, un par fenetre `nms_s`.
    levels_by_band : {nom: [dB par pas]} ; le score d'un instant est la montee maximale sur les bandes
    `full` et `crowd`. Les instants hors `windows` sont ignores."""
    bands = [levels_by_band[k] for k in ("full", "crowd") if k in levels_by_band and levels_by_band[k]]
    if not bands:
        return []
    m = min(len(b) for b in bands)
    scored = []
    for i in range(m):
        t = i * step_s
        if windows and not any(a <= t <= b for a, b in windows):
            continue
        s = max(_rise(b, i) for b in bands)
        if s > float("-inf"):
            scored.append((s, t))
    scored.sort(reverse=True)
    kept = []
    for s, t in scored:
        if all(abs(t - u) >= nms_s for _, u in kept):
            kept.append((s, t))
            if len(kept) >= n:
                break
    return sorted(t for _, t in kept)


def _team(code, home):
    return {"home": "us", "away": "opp"}.get(code, "unknown") if home else "unknown"


def extract_moments(clips, clip_s, keep, min_conf, existing, dedup_s):
    """Evenements lus dans les clips -> moments proposes (filtres, fusionnes, sans doublon avec `existing`)."""
    dropped = {"type": 0, "confidence": 0, "existing": 0, "merged": 0}
    raw = []
    for c in clips:
        if not c.get("ok"):
            continue
        ans = c.get("answer")
        evs = ans if isinstance(ans, list) else ((ans or {}).get("events") if isinstance(ans, dict) else None) or []
        for e in evs:
            if not isinstance(e, dict):
                continue
            ty = CODE.get(e.get("type"), e.get("type"))
            if ty not in keep:
                dropped["type"] += 1
                continue
            conf = float(e.get("confidence") or 0)
            if conf < min_conf:
                dropped["confidence"] += 1
                continue
            t = float(c["t"]) - clip_s / 2.0 + float(e.get("time_in_clip_s") or clip_s / 2.0)
            raw.append({"t": round(t, 1), "type": ty, "team": e.get("team") or "unknown",
                        "confidence": round(conf, 2), "note": (e.get("note") or "")[:120], "source": "clip"})
    raw.sort(key=lambda m: (m["type"], m["t"]))
    merged = []
    for m in raw:
        prev = merged[-1] if merged else None
        if prev and prev["type"] == m["type"] and abs(prev["t"] - m["t"]) < 4:
            dropped["merged"] += 1
            if m["confidence"] > prev["confidence"]:
                merged[-1] = m
            continue
        merged.append(m)
    out = []
    for m in merged:
        if any(x.get("type") == m["type"] and abs(float(x["t"]) - m["t"]) <= dedup_s for x in existing):
            dropped["existing"] += 1
            continue
        out.append(m)
    out.sort(key=lambda m: m["t"])
    return out, dropped


def _put_json(url, obj):
    """Depose le resultat complet (JSON) a une adresse d'envoi signee : la console RunPod tronque l'affichage."""
    data = json.dumps(obj).encode()
    last = None
    for k in range(4):
        try:
            req = urllib.request.Request(url, data=data, method="PUT",
                                         headers={"Content-Type": "application/json", "x-upsert": "true"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    return "upload failed: %s" % scrub(last, 120)


def sound_moments(inp):
    res = _sound_moments(inp)
    if inp.get("result_upload_url") and isinstance(res, dict):
        res["result_upload"] = _put_json(inp["result_upload_url"], res)
        if inp.get("compact_output"):
            res = {k: v for k, v in res.items() if k != "moments"} | {"n_moments": len(res.get("moments") or [])}
    return res


def _sound_moments(inp):
    t0 = time.time()
    if not inp.get("model"):
        raise RuntimeError("model is required")
    n_cand = int(inp.get("n_candidates") or 250)
    cap = int(inp.get("max_clips") or 300)
    if n_cand > cap:
        raise RuntimeError("too many candidates: %d (max_clips %d)" % (n_cand, cap))
    clip_s = int(inp.get("clip_s") or 6)
    windows = [(float(a), float(b)) for a, b in (inp.get("play_windows") or [])]
    keep = set(inp.get("keep_types") or DEFAULT_KEEP)
    min_conf = float(inp.get("min_confidence") if inp.get("min_confidence") is not None else 0.7)
    existing = inp.get("existing") or []
    dedup_s = float(inp.get("dedup_s") or 6)

    hls = HlsSource(inp["segments_json_url"]) if inp.get("segments_json_url") else None
    try:
        return _run(inp, hls, t0, n_cand, cap, clip_s, windows, keep, min_conf, existing, dedup_s)
    finally:
        if hls:
            hls.close()


def _run(inp, hls, t0, n_cand, cap, clip_s, windows, keep, min_conf, existing, dedup_s):
    tm = {}
    if hls:
        t1 = time.time()
        hls.fetch_all()
        src = hls.concat_all()
        tm["download_s"] = round(time.time() - t1, 1)
    else:
        src = inp["video_url"]
    au = audio.audio_energy({"video_url": src, "step_s": 1})
    if not au.get("has_audio"):
        return {"task": "sound_moments", "has_audio": False, "moments": [], "n_candidates": 0, "n_clips": 0}
    step = float(au.get("step_s") or 1)
    times = candidates({"full": au["full"], "crowd": au["crowd"]}, step, windows, n_cand, float(inp.get("nms_s") or 6))
    if not times:
        return {"task": "sound_moments", "has_audio": True, "moments": [], "n_candidates": 0, "n_clips": 0}

    off = int(inp.get("cand_offset") or 0)
    lim = int(inp.get("cand_limit") or len(times))
    times = times[off:off + lim]
    print("sound_moments: %d candidates in this run (offset %d), audio done after %ds" % (len(times), off, time.time() - t0), flush=True)
    cl = clipsai.clips_ai({"video_url": inp.get("video_url"), "deadline_s": inp.get("deadline_s") or 1800, "times": times, "clip_s": clip_s, "model": inp["model"],
                           "thinking_budget": inp.get("thinking_budget", 128), "home_colors": inp.get("home_colors"),
                           "away_colors": inp.get("away_colors"), "max_clips": cap, "workers": inp.get("workers") or 4},
                          cutter=hls.cutter() if hls else None)
    errors = {}
    for c in cl["clips"]:
        if not c.get("ok"):
            k = (c.get("error") or "?")[:160]
            errors[k] = errors.get(k, 0) + 1
    moments, dropped = extract_moments(cl["clips"], clip_s, keep, min_conf, existing, dedup_s)
    home = bool(inp.get("home_colors") or inp.get("away_colors"))
    for m in moments:
        m["team"] = _team(m["team"], home)
    return {"task": "sound_moments", "model": inp["model"], "has_audio": True, "n_candidates": len(times),
            "n_clips": cl["n"], "failed": cl["failed"], "errors": errors, "moments": moments, "dropped": dropped,
            "usage": cl["usage"],
            "timings": {**tm, "audio_s": (au.get("timings") or {}).get("analyse_s"), "clips_s": cl["timings"]["total_s"],
                        "total_s": round(time.time() - t0, 1)}}
