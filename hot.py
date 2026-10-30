"""Etape B du chantier « oublis de l'IA » : images denses seulement autour des
moments actifs du match, reperes par le suivi.

Pourquoi : passer de 4 s a 2 s partout (essai Argenteuil du 29/10/2026) a fait
monter le rappel de 48 % a 66 % mais double le cout et fait chuter la precision.
Ici on garde l'echantillonnage de base partout et on ajoute des images
rapprochees uniquement la ou le suivi voit de l'activite : joueurs groupes pres
d'un but, ballon pres d'un but, bloc d'equipe qui se deplace vite (transition).

Entree (apres le suivi) :
  frames    liste {"t", "points": [{"x", "y", "team", "ball"}], "calibrated"}
  windows   periodes de jeu [[debut, fin], ...] en secondes
Sortie : fenetres chaudes [[debut, fin], ...] et, si `frames_dense_upload_url`
est fourni, un zip d'images (manifest.json : frames, hot_windows, every_s)
envoye a cette adresse. Les instants deja couverts par l'echantillonnage de
base ne sont pas renvoyes.

Reglages (input ou variables d'environnement) :
  frames_dense_every_s   intervalle dans les fenetres chaudes (defaut 2 s)
  hot_share_max          part maximale du temps de jeu marquee chaude (0.35)
  hot_pad_s              marge ajoutee avant/apres chaque fenetre (4 s)
"""
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

DENSE_EVERY_S = float(os.environ.get("FRAMES_DENSE_EVERY_S", "2"))
HOT_SHARE_MAX = float(os.environ.get("HOT_SHARE_MAX", "0.35"))
HOT_PAD_S = float(os.environ.get("HOT_PAD_S", "4"))
HOT_MIN_LEN_S = float(os.environ.get("HOT_MIN_LEN_S", "6"))
HOT_MERGE_GAP_S = float(os.environ.get("HOT_MERGE_GAP_S", "6"))
SMOOTH_S = 5


def activity_per_second(frames, windows):
    """Score d'activite par seconde de jeu. Les positions sont prises telles
    quelles (metres si calees, sinon unites de l'image) : la zone « pres d'un
    but » est definie par rapport a l'etendue reelle des positions sur tout le
    match (5e–95e centiles en x), ce qui marche dans les deux cas."""
    pts = [(f["t"], p["x"], p["y"], bool(p["ball"])) for f in frames for p in f["points"]]
    if not pts:
        return {}
    xs = np.array([x for _, x, _, ball in pts if not ball], dtype=float)
    if xs.size < 50:
        return {}
    x_lo, x_hi = np.percentile(xs, 5), np.percentile(xs, 95)
    span = max(1e-6, x_hi - x_lo)
    goal_l, goal_r = x_lo + 0.25 * span, x_hi - 0.25 * span
    # Par seconde : joueurs pres d'un but, ballon pres d'un but, centre du jeu.
    per_s = {}
    for t, x, y, ball in pts:
        s = int(t)
        d = per_s.setdefault(s, {"near": 0, "ball_near": 0, "cx": [], "n": 0})
        near = x < goal_l or x > goal_r
        if ball:
            d["ball_near"] = max(d["ball_near"], int(near))
        else:
            d["near"] += int(near)
            d["cx"].append(x)
            d["n"] += 1
    n_frames_per_s = {}
    for f in frames:
        n_frames_per_s[int(f["t"])] = n_frames_per_s.get(int(f["t"]), 0) + 1
    score = {}
    prev_c = None
    for s in sorted(per_s):
        d = per_s[s]
        nf = max(1, n_frames_per_s.get(s, 1))
        near = d["near"] / nf                      # joueurs pres d'un but, par image
        c = float(np.mean(d["cx"])) if d["cx"] else None
        speed = 0.0
        if c is not None and prev_c is not None and prev_c[0] == s - 1:
            speed = abs(c - prev_c[1]) / span * 100   # deplacement du bloc, en % du terrain / s
        if c is not None:
            prev_c = (s, c)
        score[s] = near + 3.0 * d["ball_near"] + 0.5 * speed
    # Lissage sur SMOOTH_S secondes, limite aux periodes de jeu.
    out = {}
    half = SMOOTH_S // 2
    for ws, we in windows:
        for s in range(int(ws), int(we) + 1):
            vals = [score[k] for k in range(s - half, s + half + 1) if k in score]
            if vals:
                out[s] = float(np.mean(vals))
    return out


def hot_windows(frames, windows, share_max=HOT_SHARE_MAX, pad_s=HOT_PAD_S,
                min_len_s=HOT_MIN_LEN_S, merge_gap_s=HOT_MERGE_GAP_S):
    """Fenetres [debut, fin] ou l'activite depasse le seuil, au plus
    `share_max` du temps de jeu. Renvoie ([], info) si le suivi ne permet pas
    de juger (trop peu de positions)."""
    act = activity_per_second(frames, windows)
    play_s = sum(we - ws for ws, we in windows)
    if not act or play_s <= 0:
        return [], {"reason": "not enough tracking data", "seconds_scored": len(act)}
    vals = np.array(list(act.values()))
    # Au moins le double du niveau habituel : sans vrai pic, pas de fenetre.
    base_thr = max(float(np.percentile(vals, 100 * (1 - share_max))), 2.0 * float(np.median(vals)))

    def build(thr):
        hot_s = sorted(s for s, v in act.items() if v >= thr and v > 0)
        core = len(hot_s)
        spans = []
        for s in hot_s:
            if spans and s - spans[-1][1] <= merge_gap_s:
                spans[-1][1] = s
            else:
                spans.append([s, s])
        spans = [[a - pad_s, b + 1 + pad_s] for a, b in spans if (b + 1 - a) >= min_len_s]
        clipped = []
        for a, b in spans:
            for ws, we in windows:
                lo, hi = max(a, ws), min(b, we)
                if hi - lo >= min_len_s:
                    clipped.append([lo, hi])
        clipped.sort()
        merged = []
        for a, b in clipped:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return merged, core

    # Le seuil monte tant que les secondes chaudes (hors marges) depassent la
    # part maximale du temps de jeu : on garde les moments les plus actifs.
    # Si monter le seuil ne laisse plus rien, on garde le dernier resultat.
    thr = base_thr
    merged, core = build(thr)
    step = max(1e-3, 0.05 * (vals.max() - base_thr)) if vals.max() > base_thr else 1e-3
    while merged and core > share_max * play_s and thr < vals.max():
        nxt, ncore = build(thr + step)
        if not nxt:
            break
        thr += step
        merged, core = nxt, ncore
    total = sum(b - a for a, b in merged)
    info = {"threshold": round(float(thr), 3), "seconds_scored": len(act), "hot_s": round(total, 1),
            "play_s": round(play_s, 1), "share": round(total / play_s, 3), "windows": len(merged)}
    return [[round(a, 1), round(b, 1)] for a, b in merged], info


def dense_times(hot, windows, base_every, dense_every):
    """Instants a envoyer : toutes les `dense_every` s dans les fenetres chaudes,
    sauf ceux que l'echantillonnage de base (toutes les `base_every` s depuis le
    debut de chaque periode) couvre deja."""
    base = set()
    if base_every and base_every > 0:
        for ws, we in windows:
            t = ws
            while t <= we + 1e-6:
                base.add(round(t, 2))
                t += base_every
    times = []
    for a, b in hot:
        t = a
        while t <= b + 1e-6:
            tt = round(t, 2)
            if tt not in base:
                times.append(tt)
            t += dense_every
    return times


def collect_dense_frames(source, frames, windows, inp, grab_jpegs, workers=4):
    """Fenetres chaudes + zip d'images denses envoye a `frames_dense_upload_url`.
    Ne leve jamais : une erreur est rendue dans le dictionnaire."""
    url = inp.get("frames_dense_upload_url")
    hot, info = hot_windows(frames, windows,
                            share_max=float(inp.get("hot_share_max", HOT_SHARE_MAX)),
                            pad_s=float(inp.get("hot_pad_s", HOT_PAD_S)))
    out = {"hot_windows": hot, "hot_info": info, "uploaded": False, "count": 0}
    if not url or not hot or not source:
        out["reason"] = "no upload url" if not url else ("no hot window" if not hot else "no local video")
        return out
    every = float(inp.get("frames_dense_every_s", DENSE_EVERY_S))
    base_every = float(inp.get("frames_every_s", 0) or 0)
    times = dense_times(hot, windows, base_every, every)
    out["every_s"] = every
    if not times:
        out["reason"] = "nothing beyond base sampling"
        return out
    width = int(inp.get("frames_width", 960))
    quality = int(inp.get("frames_quality", 70))
    try:
        n = max(1, min(workers, len(times)))
        size = -(-len(times) // n)
        parts = [times[i:i + size] for i in range(0, len(times), size)]
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            results = list(pool.map(lambda p: grab_jpegs(source, p, width, quality), parts))
        if not any(results):
            out["reason"] = "no frame could be read from the video"
            return out
        buf = io.BytesIO()
        index = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for res in results:
                for t, jpg in res:
                    name = "frame_%09d.jpg" % int(round(t * 1000))
                    zf.writestr(name, jpg)
                    index.append({"t": round(t, 2), "file": name})
            index.sort(key=lambda d: d["t"])
            zf.writestr("manifest.json", json.dumps({"frames": index, "hot_windows": hot, "every_s": every,
                                                     "base_every_s": base_every}))
        data = buf.getvalue()
        out.update(count=len(index), bytes=len(data))
        r = requests.put(url, data=data, timeout=600, headers={"Content-Type": "application/zip", "x-upsert": "true"})
        r.raise_for_status()
        out["uploaded"] = True
    except Exception as e:
        out["error"] = str(e)[:300]
    return out
