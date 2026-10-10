"""Audit du niveau 2 (joueurs) a partir des fichiers de resultats deja enregistres.

Usage : python tools/audit_level2.py resultat1.json [resultat2.json ...]
Aucune analyse n'est lancee : on lit seulement les resultats du moteur.
"""
import json
import statistics as st
import sys
from collections import Counter


def audit(path):
    with open(path) as f:
        r = json.load(f)
    r = r.get("output", r)  # fichier brut RunPod ou resultat seul
    frames = r.get("frames") or []
    tracks = r.get("tracks") or []
    per_team = {0: [], 1: []}
    no_team, total = [], []
    for fr in frames:
        pl = [p for p in fr.get("points", []) if not p.get("ball")]
        if not pl:
            continue
        total.append(len(pl))
        no_team.append(sum(1 for p in pl if p.get("team") is None))
        for tm in (0, 1):
            per_team[tm].append(sum(1 for p in pl if p.get("team") == tm))
    med = lambda xs: round(st.median(xs), 1) if xs else None
    balance = [abs(a - b) for a, b in zip(per_team[0], per_team[1])]
    too_many = sum(1 for a, b in zip(per_team[0], per_team[1]) if a > 11 or b > 11)
    numbered = [t for t in tracks if t.get("number") is not None]
    nums = Counter((t.get("team"), t.get("number")) for t in numbered)
    # duree de vie des pistes (fragmentation du suivi)
    life = Counter()
    for fr in frames:
        for p in fr.get("points", []):
            if not p.get("ball") and p.get("trackId", -1) >= 0:
                life[p["trackId"]] += 1
    step = 0.5
    lifes_s = [n * step for n in life.values()]
    long_tracks = [tid for tid, n in life.items() if n * step >= 60]
    long_numbered = sum(1 for t in numbered if life.get(t["track_id"], 0) * step >= 60)
    return {
        "fichier": path,
        "images": len(frames),
        "joueurs visibles / image (médiane)": med(total),
        "équipe 0 / image": med(per_team[0]),
        "équipe 1 / image": med(per_team[1]),
        "sans équipe / image (arbitres, gardiens…)": med(no_team),
        "% images avec > 11 joueurs dans une équipe": round(100 * too_many / max(1, len(balance)), 1),
        "écart entre équipes (médiane)": med(balance),
        "couleurs des équipes": [c.get("hex") for c in r.get("team_colors") or []],
        "pistes": len(tracks),
        "durée de vie d'une piste (médiane, s)": med(lifes_s),
        "pistes ≥ 60 s": len(long_tracks),
        "% pistes avec numéro lu": round(100 * len(numbered) / max(1, len(tracks)), 1),
        "% pistes ≥ 60 s avec numéro lu": round(100 * long_numbered / max(1, len(long_tracks)), 1),
        "numéros distincts lus (équipe 0 / 1)": [len({n for (tm, n) in nums if tm == k}) for k in (0, 1)],
        "numéros les plus fréquents": nums.most_common(8),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(audit(p), ensure_ascii=False, indent=1))
