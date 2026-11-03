"""Ballon : le point faible du suivi (vu dans ~6 % des images sur une video Veo
a 1280 px, le ballon fait 5 a 8 px). Deux aides, independantes :

1. `zoom_detect` : quand la passe principale ne voit pas de ballon, deuxieme
   passe YOLO sur une fenetre a pleine resolution autour du groupe de joueurs
   (la camera Veo suit le ballon, il est presque toujours pres du bloc).
   Activee par BALL_ZOOM=1 (defaut : desactivee), classe 32 seule, seuil bas.
2. `interpolate` : entre deux vues du ballon proches dans le temps
   (<= BALL_GAP_S, defaut 1,5 s), position intermediaire lineaire marquee
   `seen: false`. Sert au second avis d'equipe (ball_sightings) et ne touche
   pas aux images renvoyees au site.
"""
import os

import numpy as np

BALL_ZOOM = os.environ.get("BALL_ZOOM", "0") == "1"
BALL_ZOOM_CONF = float(os.environ.get("BALL_ZOOM_CONF", "0.08"))
BALL_ZOOM_FRAC = float(os.environ.get("BALL_ZOOM_FRAC", "0.4"))    # largeur de la fenetre / largeur image
BALL_ZOOM_IMGSZ = int(os.environ.get("BALL_ZOOM_IMGSZ", "960"))
BALL_GAP_S = float(os.environ.get("BALL_GAP_S", "1.5"))


def zoom_window(player_boxes, w, h, frac=BALL_ZOOM_FRAC):
    """Fenetre (x0, y0, x1, y1) en pixels autour du centre des joueurs ;
    sans joueur, le centre de l'image. Rapport 16:9, bornee a l'image."""
    if player_boxes:
        cx = float(np.median([(b[0] + b[2]) / 2 for b in player_boxes]))
        cy = float(np.median([(b[1] + b[3]) / 2 for b in player_boxes]))
    else:
        cx, cy = w / 2, h / 2
    ww = max(320, int(w * frac))
    hh = max(180, int(ww * 9 / 16))
    x0 = int(min(max(cx - ww / 2, 0), max(w - ww, 0)))
    y0 = int(min(max(cy - hh / 2, 0), max(h - hh, 0)))
    return x0, y0, min(x0 + ww, w), min(y0 + hh, h)


def zoom_detect(model, frame, player_boxes):
    """Deuxieme passe ballon. Renvoie la boite [x1, y1, x2, y2] (pixels image
    entiere) du ballon le plus sur, ou None."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = zoom_window(player_boxes, w, h)
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    res = model.predict(frame[y0:y1, x0:x1], classes=[32], conf=BALL_ZOOM_CONF,
                        imgsz=BALL_ZOOM_IMGSZ, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.tolist()
    b = res.boxes.xyxy.tolist()[int(np.argmax(confs))]
    # un "ballon" plus grand qu'un joueur est un faux positif (panneau, tete...)
    if (b[2] - b[0]) > 0.08 * w or (b[3] - b[1]) > 0.08 * w:
        return None
    return [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]


def interpolate(frames, max_gap_s=BALL_GAP_S):
    """Liste de (t, x, y, seen) : vues reelles du ballon plus positions
    interpolees sur les images sans ballon entre deux vues distantes d'au plus
    `max_gap_s`. `frames` : [{t, points: [{x, y, ball, ...}]}] tries par t."""
    seen = []
    for f in frames:
        b = next((p for p in f["points"] if p.get("ball")), None)
        if b is not None:
            seen.append((float(f["t"]), float(b["x"]), float(b["y"])))
    if not seen:
        return []
    out = [(t, x, y, True) for t, x, y in seen]
    ts = [float(f["t"]) for f in frames]
    for (t0, x0, y0), (t1, x1, y1) in zip(seen, seen[1:]):
        if t1 - t0 <= 0 or t1 - t0 > max_gap_s:
            continue
        for t in ts:
            if t0 < t < t1:
                a = (t - t0) / (t1 - t0)
                out.append((t, x0 + a * (x1 - x0), y0 + a * (y1 - y0), False))
    out.sort(key=lambda q: q[0])
    return out
