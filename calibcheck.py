"""Controle de coherence du calage dans le temps.

Quand la camera suit le jeu, le calage automatique peut sauter d'une image a
l'autre (camera tournee vers un but, en biais, en mouvement) : tous les joueurs
semblent alors se deplacer de plusieurs metres en une seconde. On compare
chaque image calee a la derniere image gardee : si la vitesse mediane des
joueurs suivis dans les deux images depasse `max_speed` (m/s), l'image est
declaree douteuse. Un sprint de pointe fait environ 8 a 9 m/s et la mediane
normale est autour de 2 m/s, donc une mediane au-dessus de 8 m/s ne peut venir
que d'un saut du calage.

Fonction pure, sans numpy : elle ne depend que des images deja construites.
"""
from statistics import median

MAX_SPEED_MS = 8.0     # vitesse mediane maximale plausible, en m/s
MIN_DT_S = 1.0         # ecart minimal retenu pour la vitesse : a 2 images/s le bruit de suivi serait double
MIN_COMMON = 4         # joueurs suivis dans les deux images, au minimum
MAX_GAP_S = 5.0        # au-dela, on ne compare plus : l'image est acceptee


def _players(frame):
    return {p["trackId"]: (p["x"], p["y"]) for p in frame.get("points", [])
            if not p.get("ball") and p.get("trackId", -1) >= 0}


def jump_frames(frames, max_speed=MAX_SPEED_MS, min_common=MIN_COMMON, max_gap_s=MAX_GAP_S):
    """Indices des images calees dont le calage saute par rapport a la derniere
    image gardee. Une image douteuse n'est pas gardee comme reference : un
    saut isole est ecarte, un vrai changement de calage durable est masque au
    plus `max_gap_s` secondes (jusqu'a ce que la comparaison n'ait plus lieu).
    Images sans calage ignorees ; sans assez de joueurs en commun, l'image est
    acceptee (on ne peut pas juger)."""
    flagged = set()
    ref_t, ref_pts = None, None
    for i, f in enumerate(frames):
        if not f.get("calibrated"):
            continue
        t = f.get("t")
        pts = _players(f)
        if ref_t is None or t is None:
            ref_t, ref_pts = t, pts
            continue
        dt = t - ref_t
        if dt <= 0 or dt > max_gap_s:
            ref_t, ref_pts = t, pts
            continue
        common = [k for k in pts if k in ref_pts]
        if len(common) < min_common:
            ref_t, ref_pts = t, pts
            continue
        speeds = [((pts[k][0] - ref_pts[k][0]) ** 2 + (pts[k][1] - ref_pts[k][1]) ** 2) ** 0.5 / max(dt, MIN_DT_S)
                  for k in common]
        if median(speeds) > max_speed:
            flagged.add(i)
        else:
            ref_t, ref_pts = t, pts
    return flagged
