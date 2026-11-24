"""Equipes a partir des couleurs de maillot donnees par le coach.

Les couleurs « nominales » (#RRGGBB saisies sur le site) ne servent QU'A
AMORCER un regroupement en 2 groupes des couleurs reellement observees : les
maillots vus par la camera sont plus ternes et plus sombres que la couleur
du catalogue. Seuils relatifs (pas absolus). Si les couleurs manquent ou sont
trop proches, on renvoie None et le moteur garde son regroupement habituel.
Couleurs observees : [clarte, a, b] en Lab OpenCV (comme shirt_color).
"""
import re
from collections import Counter

import numpy as np
import cv2

W_LIGHT = 0.5      # poids de la clarte (la teinte compte plus : l'ombre change la clarte)
CLOSE = 15.0       # deux couleurs plus proches que cela : on ne peut pas les separer
MARGIN = 6.0       # un joueur presque a egale distance des deux groupes : sans equipe
MIN_TRACKS = 6
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def valid_hex(s):
    return isinstance(s, str) and bool(_HEX.match(s.strip()))


def hex_to_lab(s):
    s = s.strip().lstrip("#")
    bgr = np.uint8([[[int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _dist(x, c):
    d = np.asarray(x, np.float32) - np.asarray(c, np.float32)
    return float(np.sqrt((W_LIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2))


def teams_from_coach_colors(placed, hex_ours, hex_theirs):
    """Meme forme que teams_by_window : (track_team, color_team, team_rgb).
    Equipe 0 = celle du coach, 1 = adversaire. None si impossible."""
    if not (valid_hex(hex_ours) and valid_hex(hex_theirs)):
        return None
    seeds = [hex_to_lab(hex_ours), hex_to_lab(hex_theirs)]
    if _dist(seeds[0], seeds[1]) < CLOSE:
        return None
    per = {}
    for _t, pts in placed:
        for tid, _x, _y, ball, col, *_rest in pts:
            if ball or tid < 0 or col is None:
                continue
            per.setdefault(tid, []).append(np.asarray(col[:3], np.float32))
    tids = [t for t, v in per.items() if len(v) >= 3]
    if len(tids) < MIN_TRACKS:
        return None
    feats = {t: np.median(np.array(per[t]), axis=0) for t in tids}
    centers = [s.copy() for s in seeds]
    lab = {}
    for _ in range(10):
        lab = {t: int(_dist(f, centers[1]) < _dist(f, centers[0])) for t, f in feats.items()}
        new = []
        for g in (0, 1):
            m = [feats[t] for t in tids if lab[t] == g]
            new.append(np.median(np.array(m), axis=0) if m else centers[g])
        if all(np.allclose(a, b) for a, b in zip(new, centers)):
            break
        centers = new
    if _dist(centers[0], centers[1]) < CLOSE:
        return None
    d = {t: _dist(feats[t], centers[lab[t]]) for t in tids}
    limit = max(12.0, 3.0 * float(np.median(list(d.values()))))
    track_team = {}
    for t in tids:
        other = _dist(feats[t], centers[1 - lab[t]])
        if d[t] <= limit and other - d[t] >= MARGIN:
            track_team[t] = lab[t]
    if Counter(track_team.values()).most_common()[-1][1] < 3 or len(set(track_team.values())) < 2:
        return None
    return track_team, (lambda col: None), np.array(centers, np.float32)
