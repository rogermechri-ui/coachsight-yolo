"""Filtre « hors terrain » fonde sur la pelouse visible dans l'image.

Une personne dont les pieds ne sont pas sur la pelouse (public, banc, staff
derriere la ligne) n'est pas un joueur. On repere la teinte de la pelouse de
CETTE video (elle varie : vert franc, jaune-vert, ombre), on en deduit le
contour du terrain, puis on teste les pieds de chaque personne.
Bibliotheques legeres seulement (numpy, opencv) : testable sans le modele.
"""
import cv2
import numpy as np

HUE_SPREAD = 12          # demi-largeur de la teinte de pelouse autour du pic
HULL_TOL = 0.02          # tolerance du contour (part de la largeur de l'image)
STRIP_MIN = 0.35         # part minimale de pelouse autour des pieds
MIN_COVER = 0.15         # pelouse < 15 % de l'image : mesure non fiable, on ne filtre pas
OFF_MIN_FRAC = 0.5       # une piste est « hors terrain » si moins de 50 % de ses vues sont sur la pelouse
OFF_MIN_OBS = 3          # ... et si on l'a vue au moins 3 fois


def grass_mask(frame):
    """Renvoie (masque 0/255, contour convexe ou None). None = pelouse non
    trouvee de facon fiable (gros plan, salle, nuit...) : on ne filtre pas."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(64, w // 4), max(36, h // 4)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sh, sw = hsv.shape[:2]
    low = hsv[int(sh * 0.4):]
    ok = (low[..., 1] > 40) & (low[..., 2] > 30) & (low[..., 0] >= 25) & (low[..., 0] <= 95)
    if ok.sum() < 0.05 * low[..., 0].size:
        return None, None
    hist = np.bincount(low[..., 0][ok].ravel(), minlength=180)
    # lissage : pic de teinte sur 5 valeurs voisines
    sm = np.convolve(hist, np.ones(5), mode="same")
    peak = int(np.argmax(sm[25:96])) + 25
    hue = hsv[..., 0].astype(np.int16)
    mask = ((np.abs(hue - peak) <= HUE_SPREAD) & (hsv[..., 1] > 40) & (hsv[..., 2] > 30)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == big).astype(np.uint8) * 255
    if comp.sum() / 255.0 < MIN_COVER * comp.size:
        return None, None
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    hull_full = (hull.astype(np.float32) * np.array([w / sw, h / sh], np.float32)).astype(np.int32)
    return full, hull_full


def foot_on_pitch(mask, hull, box, size):
    """Les pieds (bas de la boite) sont-ils sur la pelouse ? box en pixels."""
    w, h = size
    x1, y1, x2, y2 = [float(v) for v in box]
    if cv2.pointPolygonTest(hull, (float((x1 + x2) / 2), min(y2, h - 1.0)), True) < -HULL_TOL * w:
        return False
    bh = max(2.0, y2 - y1)
    sx1, sx2 = int(max(x1, 0)), int(min(x2, w))
    sy1, sy2 = int(max(y2 - 0.1 * bh, 0)), int(min(y2 + 0.1 * bh, h))
    if sx2 <= sx1 or sy2 <= sy1:
        return True
    return float((mask[sy1:sy2, sx1:sx2] > 0).mean()) >= STRIP_MIN


def off_pitch_tracks(obs, max_removed=0.6):
    """obs : {piste: [bool sur la pelouse, ...]}. Renvoie l'ensemble des pistes
    hors terrain. Garde-fou : si le filtre retirait plus de 60 % des pistes
    (pelouse mal reperee), on ne retire rien."""
    off = {t for t, v in obs.items() if len(v) >= OFF_MIN_OBS and float(np.mean(v)) < OFF_MIN_FRAC}
    if obs and len(off) > max_removed * len(obs):
        return set()
    return off
