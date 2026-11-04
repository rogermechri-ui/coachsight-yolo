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
  video_url          adresse lisible (signee) de la video
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
  max_clips          garde-fou de cout (defaut 300) ; workers (defaut 4)
Sortie : {task, model, n_candidates, n_clips, moments:[{t, type, team, confidence, note, source:"clip"}],
          dropped:{type, confidence, existing, merged}, usage{...}, timings}
La cle Gemini est lue dans GEMINI_API_KEY (jamais dans l'entree ni le resultat).
"""
import time

import audio
import clipsai

DEFAULT_KEEP = ["goal", "shot", "big_chance", "save", "cross", "corner", "free_kick", "penalty", "foul",
                "turnover", "tackle", "counter_attack", "aerial_duel"]
# types lus par le modele -> codes de la bibliotheque d'actions du site
CODE = {"sub": "substitution"}


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
        evs = ((c.get("answer") or {}).get("events")) or []
        for e in evs:
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


def sound_moments(inp):
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

    au = audio.audio_energy({"video_url": inp["video_url"], "step_s": 1})
    if not au.get("has_audio"):
        return {"task": "sound_moments", "has_audio": False, "moments": [], "n_candidates": 0, "n_clips": 0}
    step = float(au.get("step_s") or 1)
    times = candidates({"full": au["full"], "crowd": au["crowd"]}, step, windows, n_cand, float(inp.get("nms_s") or 6))
    if not times:
        return {"task": "sound_moments", "has_audio": True, "moments": [], "n_candidates": 0, "n_clips": 0}

    cl = clipsai.clips_ai({"video_url": inp["video_url"], "times": times, "clip_s": clip_s, "model": inp["model"],
                           "thinking_budget": inp.get("thinking_budget", 128), "home_colors": inp.get("home_colors"),
                           "away_colors": inp.get("away_colors"), "max_clips": cap, "workers": inp.get("workers") or 4})
    moments, dropped = extract_moments(cl["clips"], clip_s, keep, min_conf, existing, dedup_s)
    home = bool(inp.get("home_colors") or inp.get("away_colors"))
    for m in moments:
        m["team"] = _team(m["team"], home)
    return {"task": "sound_moments", "model": inp["model"], "has_audio": True, "n_candidates": len(times),
            "n_clips": cl["n"], "failed": cl["failed"], "moments": moments, "dropped": dropped,
            "usage": cl["usage"],
            "timings": {"audio_s": (au.get("timings") or {}).get("analyse_s"), "clips_s": cl["timings"]["total_s"],
                        "total_s": round(time.time() - t0, 1)}}
