"""Calage du terrain image par image, ajuste sur les lignes blanches.

Principe : a partir d'un calage approximatif (homographie terrain -> image),
on redessine les lignes du terrain sur l'image et on deplace legerement ce
dessin jusqu'a ce qu'il colle aux lignes blanches reellement visibles sur la
pelouse. Le score final dit quelle part du dessin tombe sur une vraie ligne.
"""
import cv2
import numpy as np
from scipy.optimize import minimize

# Gabarit en metres (meme convention que PITCH_KEYPOINTS dans handler.py :
# x le long du terrain depuis le but de gauche, y depuis la touche du haut).
# Taille par defaut : 105 x 68 (mesuree sur Frock Field, Catawba) ; les
# marquages (surfaces, rond central, point de penalty) ont des tailles fixes
# par le reglement, quelle que soit la taille du terrain.
L, W = 105.0, 68.0
PBW, PBL, GBW, GBL, CCR, PSD = 40.32, 16.5, 18.32, 5.5, 9.15, 11.0


def pitch_polylines(length=L, width=W):
    """Lignes du terrain en metres, sous forme de polylignes."""
    y1, y2 = (width - PBW) / 2, (width + PBW) / 2
    g1, g2 = (width - GBW) / 2, (width + GBW) / 2
    lines = [
        [(0, 0), (length, 0)], [(0, width), (length, width)],
        [(0, 0), (0, width)], [(length, 0), (length, width)],
        [(length / 2, 0), (length / 2, width)],
        [(0, y1), (PBL, y1), (PBL, y2), (0, y2)],
        [(length, y1), (length - PBL, y1), (length - PBL, y2), (length, y2)],
        [(0, g1), (GBL, g1), (GBL, g2), (0, g2)],
        [(length, g1), (length - GBL, g1), (length - GBL, g2), (length, g2)],
    ]
    a = np.linspace(0, 2 * np.pi, 73)
    lines.append([(length / 2 + CCR * np.cos(t), width / 2 + CCR * np.sin(t)) for t in a])
    return lines


def sample_pitch(step=0.5, length=L, width=W):
    """Points regulierement espaces (tous les `step` metres) sur les lignes."""
    pts = []
    for pl in pitch_polylines(length, width):
        for (x0, y0), (x1, y1) in zip(pl[:-1], pl[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
            for k in range(n):
                f = k / n
                pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    return np.float32(pts)


def line_mask(img):
    """Pixels des lignes blanches sur la pelouse (et masque de la pelouse)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 30), (95, 255, 255))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    white = ((tophat > 25) & (hsv[:, :, 1] < 80)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, grass)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # On ne garde que les traces fines et longues (lignes) : les maillots
    # blancs, le ballon et les reflets forment des taches epaisses qu'on ecarte.
    # On efface les zones epaisses (et leurs abords) pixel par pixel, ce qui
    # garde le reste d'un reseau de lignes meme si un croisement est epais,
    # puis on ne garde que les traces assez longues.
    dt = cv2.distanceTransform(white, cv2.DIST_L2, 3)
    thick = (dt > 4.5).astype(np.uint8)
    if thick.any():
        grow = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        white[grow > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    if n > 1:
        span = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep = np.zeros(n, bool)
        keep[1:] = span >= 30
        white = (keep[lab] * 255).astype(np.uint8)
    return white, grass


def content_box(img, thresh=16):
    """Zone utile de l'image (sans les bandes noires sur les cotes)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = np.where(g.mean(axis=0) > thresh)[0]
    rows = np.where(g.mean(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class LineFitter:
    """Ajuste une homographie terrain -> image sur les lignes blanches."""

    def __init__(self, img, length=L, width=W, trunc=12.0):
        self.h, self.w = img.shape[:2]
        self.white, self.grass = line_mask(img)
        inv = 255 - self.white
        self.dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        self.trunc = trunc
        self.length, self.width = length, width
        self.samples = sample_pitch(0.5, length, width)

    def _project(self, Hpi):
        p = cv2.perspectiveTransform(self.samples.reshape(-1, 1, 2), Hpi).reshape(-1, 2)
        return p

    def cost(self, Hpi):
        p = self._project(Hpi)
        x, y = p[:, 0], p[:, 1]
        ok = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        if ok.sum() < 30:
            return 1e3, 0.0, int(ok.sum())
        xi, yi = x[ok].astype(int), y[ok].astype(int)
        on_grass = self.grass[yi, xi] > 0
        if on_grass.sum() < 30:
            return 1e3, 0.0, int(on_grass.sum())
        d = np.minimum(self.dist[yi[on_grass], xi[on_grass]], self.trunc)
        hit = float((d <= 3).mean())
        return float(d.mean()), hit, int(on_grass.sum())

    def refine(self, Hpi, iters=2):
        """Hpi : homographie terrain (m) -> image (px). Renvoie (H, score)."""
        # Parametrage par la position image de 4 points de controle du terrain
        # (bien conditionne) ; on part des points visibles les plus ecartes.
        L_, W_ = self.length, self.width
        ctrl = np.float32([[L_ * 0.25, W_ * 0.25], [L_ * 0.75, W_ * 0.25],
                           [L_ * 0.75, W_ * 0.75], [L_ * 0.25, W_ * 0.75]])
        p0 = cv2.perspectiveTransform(ctrl.reshape(-1, 1, 2), Hpi).reshape(-1)

        def to_H(p):
            return cv2.getPerspectiveTransform(ctrl, np.float32(p).reshape(4, 2))

        n0 = max(30, self.cost(Hpi)[2])

        def f(p):
            # Distance moyenne aux lignes + penalite si le dessin "sort" de la
            # pelouse (sinon l'optimiseur pourrait tricher en ne gardant que
            # quelques points chanceux).
            try:
                d, _, n = self.cost(to_H(p))
            except cv2.error:
                return 1e3
            return d + self.trunc * max(0.0, 1.0 - n / n0) * 2.0

        # Du grossier au fin : distance tronquee large d'abord (bassin
        # d'attraction large), puis de plus en plus fine (precision au pixel).
        best = p0
        keep = self.trunc
        for trunc, scale in ((60.0, 40.0), (25.0, 15.0), (10.0, 5.0), (5.0, 2.0)):
            self.trunc = trunc
            res = minimize(f, best, method="Powell",
                           options={"xtol": 0.2, "ftol": 1e-4, "maxfev": 6000,
                                    "direc": np.eye(8) * scale})
            best = res.x
        self.trunc = keep
        H = to_H(best)
        mean_d, hit, n = self.cost(H)
        return H, {"mean_dist_px": round(mean_d, 2), "line_hit": round(hit, 3), "samples": n}


def homography_from_keypoints(kps, template, min_conf=0.5):
    """kps : [(id, x, y, conf)] en pixels ; template : [(X, Y)] en metres.
    Renvoie l'homographie terrain -> image (RANSAC) ou None."""
    good = [k for k in kps if k[3] >= min_conf and 0 <= k[0] < len(template)]
    if len(good) < 4:
        return None, 0
    src = np.float32([template[c] for c, _, _, _ in good])
    dst = np.float32([[x, y] for _, x, y, _ in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if H is None:
        return None, 0
    return H, int(mask.sum())


def draw_overlay(img, Hpi, color=(0, 255, 255), length=L, width=W):
    out = img.copy()
    for pl in pitch_polylines(length, width):
        p = cv2.perspectiveTransform(np.float32([pl]), Hpi)[0]
        if np.all(np.abs(p) < 1e5):
            cv2.polylines(out, [p.astype(np.int32)], False, color, 2, cv2.LINE_AA)
    return out


def refine_icp(fitter, Hpi, radii=(80, 50, 30, 18, 10, 6, 4, 3), rounds=3):
    """Ajustement iteratif (type ICP) : chaque point du dessin est rapproche de
    la ligne blanche la plus proche, puis on recalcule l'homographie ; le rayon
    de recherche diminue progressivement. Plus robuste que l'optimisation
    directe quand le calage de depart est approximatif."""
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero(fitter.white)
    if len(xs) < 50:
        return Hpi, {"mean_dist_px": None, "line_hit": 0.0, "samples": 0}
    tree = cKDTree(np.c_[xs, ys].astype(np.float32))
    H = Hpi.copy()
    for r in radii:
        for _ in range(rounds):
            p = cv2.perspectiveTransform(fitter.samples.reshape(-1, 1, 2), H).reshape(-1, 2)
            ok = (p[:, 0] >= 0) & (p[:, 0] < fitter.w) & (p[:, 1] >= 0) & (p[:, 1] < fitter.h)
            if ok.sum() < 20:
                break
            idx = np.where(ok)[0]
            gi = fitter.grass[p[idx, 1].astype(int), p[idx, 0].astype(int)] > 0
            idx = idx[gi]
            if len(idx) < 20:
                break
            d, j = tree.query(p[idx], distance_upper_bound=r)
            m = np.isfinite(d)
            if m.sum() < 12:
                break
            src = fitter.samples[idx[m]]
            dst = tree.data[j[m]]
            Hn, mask = cv2.findHomography(src, dst, cv2.RANSAC, max(2.0, r / 3))
            if Hn is None:
                break
            H = Hn
    mean_d, hit, n = fitter.cost(H)
    return H, {"mean_dist_px": round(mean_d, 2), "line_hit": round(hit, 3), "samples": n}


def fit_best(img, Hpi, restarts=12, seed=0, length=L, width=W):
    """Meilleur calage a partir d'un calage approximatif : ajustement direct,
    puis plusieurs departs legerement decales ; on garde celui qui colle le
    mieux aux lignes blanches (part du dessin sur une vraie ligne)."""
    rng = np.random.default_rng(seed)
    fitter = LineFitter(img, length, width)
    ctrl = np.float32([[length * 0.25, width * 0.25], [length * 0.75, width * 0.25],
                       [length * 0.75, width * 0.75], [length * 0.25, width * 0.75]])
    base = cv2.perspectiveTransform(ctrl.reshape(-1, 1, 2), Hpi).reshape(-1, 2)
    starts = [Hpi]
    try:
        starts.append(fitter.refine(Hpi)[0])
    except Exception:
        pass
    for _ in range(restarts):
        p = base + rng.normal(0, 45, base.shape)
        try:
            starts.append(cv2.getPerspectiveTransform(ctrl, np.float32(p)))
        except cv2.error:
            pass
    n_ref = max(30, fitter.cost(Hpi)[2])

    def scale_at_center(H):
        """Taille apparente (px) d'un carre de 10 m au centre de l'image."""
        try:
            Hi = np.linalg.inv(H)
            c = cv2.perspectiveTransform(np.float32([[[fitter.w / 2, fitter.h * 0.6]]]), Hi)[0, 0]
            q = np.float32([[c, c + [10, 0], c + [0, 10]]])
            p = cv2.perspectiveTransform(q, H)[0]
            return float(np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2
        except Exception:
            return None

    s_ref = scale_at_center(Hpi)
    best = None
    for H0 in starts:
        try:
            H, sc = refine_icp(fitter, H0)
        except Exception:
            continue
        # Garde-fous contre les solutions degenerees (terrain "ecrase" sur une
        # seule ligne) : taille apparente et couverture proches du depart.
        s = scale_at_center(H)
        if s_ref and (s is None or not (0.6 < s / s_ref < 1.7)):
            continue
        if not (0.5 < sc["samples"] / n_ref < 1.6):
            continue
        key = (sc["line_hit"], -(sc["mean_dist_px"] or 99))
        if best is None or key > best[0]:
            best = (key, H, sc)
    return (best[1], best[2]) if best else (Hpi, {"mean_dist_px": None, "line_hit": 0.0, "samples": 0})
