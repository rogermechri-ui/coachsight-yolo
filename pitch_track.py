"""Calage du terrain image par image pour une camera Veo (camera fixe).

Les vues "suivi" de Veo sont des recadrages virtuels d'une seule camera fixe,
posee au niveau de la ligne mediane derriere la ligne de touche. Une image est
donc entierement decrite par la position de la camera C (une fois par stade)
et trois reglages : cap (gauche-droite), inclinaison (haut-bas) et focale (zoom).

Suivi :
- mouvement de la camera d'une image a l'autre mesure sur le decor fixe
  (arbres, tribunes, panneaux) : marche meme quand aucune ligne n'est visible ;
- recalage fin sur les lignes blanches de la pelouse ;
- recherche globale (cap, inclinaison, zoom) pour demarrer ou se raccrocher.

Repere terrain : x le long du terrain (0..L), y depuis la touche opposee a la
camera (0..W) ; la camera est du cote y = W.
"""
import time

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

import pitch_fit as pf

WORK_W = 1280          # largeur de travail (les videos 3440 px sont reduites)


# --- Pelouse et lignes blanches -------------------------------------------------

def field_mask(img, gap=14):
    """Pelouse : pour chaque colonne, on remonte depuis le bas tant que c'est de
    l'herbe (trous de moins de `gap` px toleres : lignes, joueurs), puis on lisse
    la limite haute (les arbres au-dessus des panneaux ne comptent pas)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (18, 40, 40), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)) > 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid = gray[h - 60:h - 20].mean(0) > 25       # pas les bandes noires
    xs = np.where(valid)[0]
    f = np.zeros((h, w), np.uint8)
    if not len(xs):
        return f
    # Premiere rupture (plus de `gap` px sans herbe) en remontant depuis le bas.
    gg = g[::-1, xs]                                 # du bas vers le haut
    miss = np.zeros(len(xs), int)
    top = np.full(len(xs), h - 1)
    alive = np.ones(len(xs), bool)
    for r in range(h):
        row = gg[r]
        miss = np.where(row, 0, miss + 1)
        top = np.where(alive & row, h - 1 - r, top)
        alive &= miss <= gap
        if not alive.any():
            break
    top = median_filter(top, size=61, mode='nearest').astype(int)
    rows = np.arange(h)[:, None]
    f[:, xs] = (rows >= top[None, :]).astype(np.uint8) * 255
    return f


def line_mask(img, field):
    h = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ridge = g8.astype(np.float32) - cv2.medianBlur(g8, 11).astype(np.float32)
    rows = np.arange(h)[:, None] / h
    ridge2 = np.zeros_like(ridge)
    y6 = int(h * 0.6)
    ridge2[y6:] = g8[y6:].astype(np.float32) - cv2.medianBlur(g8[y6 - 15:], 31)[15:].astype(np.float32)
    fm = field > 0
    mad = float(np.median(np.abs(ridge[fm]))) if fm.any() else 3.0
    thr = max(5.0, 4.5 * mad)
    white = (((ridge > thr) | ((ridge2 > 18) & (rows > 0.6))) & (hsv[:, :, 1] < 120)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, cv2.erode(field, np.ones((5, 5), np.uint8)))
    # Epaisseur toleree croissante vers le bas (perspective) : les joueurs et
    # le ballon (taches epaisses) sont ecartes, les lignes gardees.
    dt = cv2.distanceTransform(white, cv2.DIST_L2, 3)
    thick = (dt > 2.5 + 6.0 * rows).astype(np.uint8)
    if thick.any():
        white[cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    if n > 1:
        span = np.maximum(st[1:, 2], st[1:, 3])
        keep = np.zeros(n, bool)
        keep[1:] = (span >= 25) & (st[1:, 4] < span * 10)
        white = (keep[lab] * 255).astype(np.uint8)
    return white


# --- Camera ------------------------------------------------------------------------

def homog(C, th, ph, f, cx, cy):
    """Homographie terrain (x, y, 0) -> image, camera en C, cap th, inclinaison ph, focale f."""
    fw = np.array([np.sin(th) * np.cos(ph), -np.cos(th) * np.cos(ph), -np.sin(ph)])
    rt = np.array([np.cos(th), np.sin(th), 0.0])
    dn = np.cross(rt, fw)
    R = np.stack([rt, dn, fw])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    t = -R @ np.asarray(C, float)
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def homog_batch(C, P, cx, cy):
    """P : (N, 3) = cap, inclinaison, focale. Renvoie (N, 3, 3)."""
    th, ph, f = P[:, 0], P[:, 1], P[:, 2]
    fw = np.stack([np.sin(th) * np.cos(ph), -np.cos(th) * np.cos(ph), -np.sin(ph)], 1)
    rt = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    dn = np.cross(rt, fw)
    R = np.stack([rt, dn, fw], 1)                                  # (N,3,3)
    K = np.zeros((len(P), 3, 3)); K[:, 0, 0] = f; K[:, 1, 1] = f
    K[:, 0, 2] = cx; K[:, 1, 2] = cy; K[:, 2, 2] = 1
    t = -R @ np.asarray(C, float)
    M = np.concatenate([R[:, :, :2], t[:, :, None]], 2)
    return K @ M


class PitchModel:
    """Lignes du terrain : points tous les metres, polylignes denses pour le
    dessin, et carte des distances a la ligne la plus proche (en metres)."""
    RES, PAD = 0.1, 25.0

    def __init__(self, L, W):
        self.L, self.W = L, W
        s = pf.sample_pitch(1.0, L, W).astype(np.float64)
        self.samples = np.column_stack([s, np.ones(len(s))])
        self.dense = []
        for pl in pf.pitch_polylines(L, W):
            pl = np.asarray(pl, float)
            seg = []
            for a, b in zip(pl[:-1], pl[1:]):
                n = max(2, int(np.hypot(*(b - a)) / 0.5))
                seg.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
            seg.append(pl[-1:])
            self.dense.append(np.vstack(seg))
        gw, gh = int((L + 2 * self.PAD) / self.RES), int((W + 2 * self.PAD) / self.RES)
        canvas = np.zeros((gh, gw), np.uint8)
        for pl in pf.pitch_polylines(L, W):
            p = ((np.asarray(pl) + self.PAD) / self.RES).astype(np.int32)
            cv2.polylines(canvas, [p], False, 255, 1)
        self.dt = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 5) * self.RES
        self.gw, self.gh = gw, gh

    def draw(self, img, H, color=(0, 255, 255), thick=2):
        out = img.copy()
        for pl in self.dense:
            q = np.column_stack([pl, np.ones(len(pl))]) @ H.T
            good = q[:, 2] > 0.5
            p = q[:, :2] / np.where(good, q[:, 2], 1)[:, None]
            for i in range(len(pl) - 1):
                if good[i] and good[i + 1] and np.all(np.abs(p[i:i + 2]) < 2e4):
                    cv2.line(out, tuple(np.int32(p[i])), tuple(np.int32(p[i + 1])), color, thick)
        return out


class View:
    """Une image prete pour le calage (a la largeur de travail)."""

    def __init__(self, img, box=None, n_white=300, seed=0):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.field = field_mask(img)
        self.white = line_mask(img, self.field)
        x0, y0, x1, y1 = box or pf.content_box(img)
        self.box = (x0, y0, x1, y1)
        self.white[self.h - 70:, max(0, x1 - 130):] = 0     # logo veo
        self.white[self.h - 70:, :x0 + 40] = 0                 # icone du lecteur
        self.dt = cv2.distanceTransform(255 - self.white, cv2.DIST_L2, 5)
        w = np.column_stack(np.where(self.white > 0))[:, ::-1].astype(np.float64)
        if len(w) > n_white:
            w = w[np.random.default_rng(seed).choice(len(w), n_white, replace=False)]
        self.wpts = np.column_stack([w, np.ones(len(w))]) if len(w) else np.zeros((0, 3))
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def score_batch(v, model, Hs, T=15.0, Tm=1.5, min_n=40):
    """Ecart entre le terrain dessine et les lignes vues (0 = parfait, ~2 = rien ne colle) :
    dessin -> lignes blanches (pixels) + lignes blanches -> terrain (metres)."""
    P = np.einsum('nij,mj->nmi', Hs, model.samples)
    z = P[:, :, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = P[:, :, 0] / z
        vv = P[:, :, 1] / z
    x0, y0, x1, y1 = v.box
    ok = (z > 0) & (u >= x0) & (u < x1 - 1) & (vv >= 0) & (vv < v.h - 1)
    ui = np.clip(np.nan_to_num(u), 0, v.w - 1).astype(int)
    vi = np.clip(np.nan_to_num(vv), 0, v.h - 1).astype(int)
    onf = v.field[vi, ui] > 0
    d = np.where(onf, np.minimum(v.dt[vi, ui], T), T)    # trait hors pelouse = penalite maximale
    n = ok.sum(1)
    c1 = np.where((ok & onf).sum(1) >= min_n, (d * ok).sum(1) / np.maximum(n, 1), T)
    if len(v.wpts) == 0:
        c2 = np.full(len(Hs), Tm)
    else:
        Hi = np.linalg.inv(Hs)
        Q = np.einsum('nij,mj->nmi', Hi, v.wpts)
        zq = Q[:, :, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            gx = (Q[:, :, 0] / zq + model.PAD) / model.RES
            gy = (Q[:, :, 1] / zq + model.PAD) / model.RES
        okq = (zq > 0) & (gx >= 0) & (gx < model.gw - 1) & (gy >= 0) & (gy < model.gh - 1)
        dd = model.dt[np.clip(np.nan_to_num(gy), 0, model.gh - 1).astype(int),
                      np.clip(np.nan_to_num(gx), 0, model.gw - 1).astype(int)]
        d2 = np.minimum(np.where(okq, dd, Tm), Tm)
        # moyenne sur les 70 % de pixels blancs les mieux expliques : les joueurs,
        # panneaux et spectateurs pris pour des lignes ne penalisent pas le bon calage
        k = max(1, int(d2.shape[1] * 0.7))
        c2 = np.partition(d2, k - 1, axis=1)[:, :k].mean(1)
    return c1 / T + c2 / Tm


# Plages realistes d'une vue Veo (image de travail de 1280 px) : zoom et inclinaison.
F_RANGE = (450.0, 2600.0)
F_TILT = (-1.0, 16.0)


def _plausible(p):
    return F_RANGE[0] * 0.85 <= p[2] <= F_RANGE[1] * 1.2 and np.radians(F_TILT[0] - 2) <= p[1] <= np.radians(F_TILT[1] + 4)


def _grid(p, dp, n):
    th = p[0] + np.radians(np.linspace(-dp[0], dp[0], n[0]))
    ph = p[1] + np.radians(np.linspace(-dp[1], dp[1], n[1]))
    f = p[2] * np.linspace(1 - dp[2], 1 + dp[2], n[2])
    return np.array(np.meshgrid(th, ph, f, indexing='ij')).reshape(3, -1).T


def local_search(v, model, C, p, dp=(1.0, 0.4, 0.04), cx=640.0, cy=268.0):
    """Ajustement fin autour de p : grille, puis grille resserree."""
    best_p, best_s = np.asarray(p, float), None
    for k in range(2):
        G = _grid(best_p, dp, (11, 5, 5))
        G = G[(G[:, 2] >= F_RANGE[0] * 0.85) & (G[:, 2] <= F_RANGE[1] * 1.2)]
        if not len(G):
            break
        s = score_batch(v, model, homog_batch(C, G, cx, cy))
        i = int(np.argmin(s))
        if best_s is None or s[i] < best_s:
            best_p, best_s = G[i], float(s[i])
        dp = (dp[0] / 3, dp[1] / 3, dp[2] / 3)
    return best_p, best_s


def global_search(v, model, C, cx=640.0, cy=268.0, keep=16):
    """Recherche sur tous les reglages possibles (demarrage, ou camera perdue)."""
    th = np.radians(np.arange(-100, 100.1, 3.0))
    ph = np.radians(np.arange(F_TILT[0], F_TILT[1] + 0.1, 1.0))
    fs = np.geomspace(F_RANGE[0], F_RANGE[1], 13)
    cands = []
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    base = np.column_stack([TH.ravel(), PH.ravel()])
    for f in fs:
        P = np.column_stack([base, np.full(len(base), f)])
        s = score_batch(v, model, homog_batch(C, P, cx, cy))
        for i in np.argsort(s)[:4]:
            cands.append((float(s[i]), P[i]))
    cands.sort(key=lambda c: c[0])
    best = None
    for s, p in cands[:keep]:
        q, sq = local_search(v, model, C, p, dp=(2.0, 1.0, 0.08), cx=cx, cy=cy)
        if best is None or sq < best[1]:
            best = (q, sq)
    return best


# --- Mouvement de la camera entre deux images ---------------------------------------

_orb = cv2.ORB_create(1000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


class Feat:
    """Points de decor (demi-resolution) pour mesurer le mouvement de la camera."""

    def __init__(self, gray, box):
        h, w = gray.shape
        x0, y0, x1, y1 = box
        m = np.zeros((h, w), np.uint8)
        m[:, x0 + 12:x1 - 12] = 255
        m[h - 70:, max(0, x1 - 140):] = 0
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(m, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        self.k, self.d = _orb.detectAndCompute(small, ms)


_S = np.diag([2.0, 2.0, 1.0])
_Si = np.diag([0.5, 0.5, 1.0])


def camera_motion(fa, fb):
    """Homographie image a -> image b (pixels de l'image de travail)."""
    if fa.d is None or fb.d is None or len(fa.k) < 20 or len(fb.k) < 20:
        return None, 0
    m = _bf.match(fa.d, fb.d)
    if len(m) < 20:
        return None, 0
    pa = np.float32([fa.k[x.queryIdx].pt for x in m])
    pb = np.float32([fb.k[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5)
    if H is None:
        return None, 0
    return _S @ H @ _Si, int(inl.sum())


def fit_params(C, Hpred, p0, cx, cy, h):
    """Cap / inclinaison / focale qui reproduisent le mieux une homographie predite."""
    grid = np.float64([[x, y] for x in np.linspace(cx - 440, cx + 440, 9)
                       for y in np.linspace(h * 0.5, h * 0.97, 5)])
    Q = np.column_stack([grid, np.ones(len(grid))]) @ np.linalg.inv(Hpred).T
    ok = Q[:, 2] > 0
    if ok.sum() < 6:
        return np.asarray(p0, float)
    X = Q[ok, :2] / Q[ok, 2:3]
    U = grid[ok]

    def res(x):
        H = homog(C, x[0], x[1], x[2] * 1000, cx, cy)
        q = np.column_stack([X, np.ones(len(X))]) @ H.T
        return ((q[:, :2] / q[:, 2:3]) - U).ravel()
    r = least_squares(res, [p0[0], p0[1], p0[2] / 1000])
    return np.array([r.x[0], r.x[1], r.x[2] * 1000])


# --- Suivi --------------------------------------------------------------------------

LOCK_SCORE = 0.95      # score maxi pour demarrer le suivi sur une image
MIN_INLIERS = 30       # points de decor coherents pour accepter un mouvement


def to_work(img):
    """Image reduite a la largeur de travail."""
    h, w = img.shape[:2]
    if w == WORK_W:
        return img
    return cv2.resize(img, (WORK_W, int(round(h * WORK_W / w))), interpolation=cv2.INTER_AREA)


class PitchTracker:
    """Suivi du calage, images fournies dans l'ordre (a la largeur de travail) :
    - motion(img) : image intermediaire, seul le mouvement de la camera est suivi
      (rapide, a appeler plusieurs fois par seconde pendant les panoramiques) ;
    - full(img) : image analysee, mouvement + recalage sur les lignes ; renvoie
      l'homographie terrain -> image (pixels de travail) ou None, et le score."""

    def __init__(self, camera, L, W, verify_every=30, global_every=8):
        self.C = np.asarray(camera, float)
        self.model = PitchModel(L, W)
        self.p = None
        self.feat = None
        self.box = None
        self.h = None
        self.lost = False
        self.full_n = 0
        self.last_global = -10 ** 9
        self.verify_every = verify_every
        self.global_every = global_every
        self.stats = {"frames": 0, "calibrated": 0, "motion_frames": 0, "motion_failed": 0,
                      "global_searches": 0, "relocks": 0, "time_s": 0.0}

    def _center(self):
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, self.h / 2.0

    def _move(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.box is None:
            self.box = pf.content_box(img)
            self.h = img.shape[0]
        f = Feat(gray, self.box)
        if self.p is not None and self.feat is not None:
            Hr, inl = camera_motion(self.feat, f)
            if Hr is not None and inl >= MIN_INLIERS:
                cx, cy = self._center()
                Hp = Hr @ homog(self.C, *self.p[:2], self.p[2], cx, cy)
                q = fit_params(self.C, Hp, self.p, cx, cy, self.h)
                if _plausible(q):
                    self.p = q
                else:                     # reglage invraisemblable : on se recalera
                    self.stats["motion_failed"] += 1
                    self.lost = True
            else:
                self.stats["motion_failed"] += 1
                self.lost = True        # mouvement non mesure : on se recalera sur les lignes
        self.feat = f

    def motion(self, img):
        t0 = time.time()
        self._move(img)
        self.stats["motion_frames"] += 1
        self.stats["time_s"] += time.time() - t0

    def current(self):
        """Homographie actuelle (apres le dernier mouvement suivi), ou None."""
        if self.p is None or self.box is None:
            return None
        self.stats["frames"] += 1
        self.stats["calibrated"] += 1
        cx, cy = self._center()
        return homog(self.C, *self.p[:2], self.p[2], cx, cy)

    def _global(self, v, cx, cy):
        self.last_global = self.full_n
        self.stats["global_searches"] += 1
        t = time.time()
        g = global_search(v, self.model, self.C, cx, cy)
        self.stats["global_s"] = self.stats.get("global_s", 0.0) + time.time() - t
        return g

    def full(self, img):
        t0 = time.time()
        self._move(img)
        self.full_n += 1
        v = View(img, self.box)
        cx, cy = self._center()
        score = None
        if self.p is not None:
            s0 = float(score_batch(v, self.model, homog(self.C, *self.p[:2], self.p[2], cx, cy)[None])[0])
            dp = (1.0, 0.4, 0.04) if (s0 < 0.9 and not self.lost) else (5.0, 2.0, 0.2)
            q, s1 = local_search(v, self.model, self.C, self.p, dp, cx, cy)
            if s1 < s0 - 0.02 and (s1 < LOCK_SCORE or s1 < s0 - 0.15):
                self.p, score = q, s1
            else:
                score = s0
            self.lost = False
            # Verification de temps en temps : recherche complete gardee seulement
            # si elle colle nettement mieux.
            if score > 1.4 and self.full_n - self.last_global >= self.verify_every:
                g = self._global(v, cx, cy)
                if g and g[1] < 1.0 and g[1] < score - 0.3:
                    self.p, score = g[0], g[1]
                    self.stats["relocks"] += 1
        elif self.full_n - self.last_global >= self.global_every or self.last_global < 0:
            g = self._global(v, cx, cy)
            if g and g[1] < LOCK_SCORE:
                self.p, score = g[0], g[1]
        self.stats["frames"] += 1
        self.stats["time_s"] += time.time() - t0
        if self.p is None:
            return None, None
        self.stats["calibrated"] += 1
        return homog(self.C, *self.p[:2], self.p[2], cx, cy), score


def image_to_pitch(H, u, v, w, h):
    """(u, v) normalises (0..1) dans l'image -> metres sur le terrain."""
    q = np.linalg.solve(H, np.array([u * w, v * h, 1.0]))
    if q[2] == 0:
        return None
    return q[0] / q[2], q[1] / q[2]


# --- Position de la camera : estimation automatique --------------------------------

CAMERA_PRIOR = (0.5, 4.0, 4.5)   # (part de L, metres derriere la touche, hauteur) : pose Veo typique
CAMERA_MAX_VIEWS = 60
CAMERA_MAX_SCORE = 1.1            # au-dela, la position trouvee n'est pas retenue
CAMERA_HEIGHTS = (2.5, 4.5, 7.0)  # departs en hauteur (m)


def _bilinear(img, x, y):
    x = np.clip(x, 0, img.shape[1] - 1.001)
    y = np.clip(y, 0, img.shape[0] - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _joint_residuals(x, views, model, cx, cy):
    """Ecarts lignes vues <-> terrain pour toutes les vues, derivables (interpolation)."""
    C = x[:3]
    P = x[3:].reshape(-1, 3) * np.array([1.0, 1.0, 1000.0])
    out = []
    for k, v in enumerate(views):
        H = homog(C, P[k, 0], P[k, 1], P[k, 2], cx, cy)
        Hi = np.linalg.inv(H)
        Q = v.wpts @ Hi.T
        z = Q[:, 2]
        with np.errstate(all='ignore'):
            gx = np.nan_to_num((Q[:, 0] / z + model.PAD) / model.RES)
            gy = np.nan_to_num((Q[:, 1] / z + model.PAD) / model.RES)
        okq = (z > 0) & (gx >= 0) & (gx < model.gw - 1) & (gy >= 0) & (gy < model.gh - 1)
        d = np.minimum(np.where(okq, _bilinear(model.dt, gx, gy), 1.5), 1.5)
        out.append(d / 1.5 / np.sqrt(max(1, len(d))))
        S = model.samples @ H.T
        zs = S[:, 2]
        with np.errstate(all='ignore'):
            u = np.nan_to_num(S[:, 0] / zs)
            vv = np.nan_to_num(S[:, 1] / zs)
        x0, y0, x1, y1 = v.box
        ok = (zs > 0) & (u >= x0) & (u < x1 - 1) & (vv >= 0) & (vv < v.h - 1)
        ui = np.clip(u, 0, v.w - 1).astype(int)
        vi = np.clip(vv, 0, v.h - 1).astype(int)
        onf = v.field[vi, ui] > 0
        dd = np.where(ok & onf, np.minimum(_bilinear(v.dt, u, vv), 15.0), 15.0)
        out.append(np.where(ok, dd, 0.0) / 15.0 / np.sqrt(max(1, ok.sum())))
    return np.concatenate(out)


def mean_score(views, model, C, P, cx, cy):
    return float(np.mean([score_batch(v, model, homog_batch(C, P[k:k + 1], cx, cy))[0]
                          for k, v in enumerate(views)]))


def estimate_camera(frames, L, W, prior=None, max_views=CAMERA_MAX_VIEWS, log=print):
    """Position de la camera a partir d'images d'un match (camera fixe type Veo).

    frames : iterateur d'images BGR (pleine taille ou de travail), dans l'ordre,
    environ 6 par seconde de video, sur un ou plusieurs passages du match.
    1. Suivi du terrain avec une position a priori (ligne mediane, 4 m derriere
       la touche, 4,5 m de haut) ; une image sur trois est recalee sur les lignes.
    2. Les images bien calees (jusqu'a max_views, reparties) servent a ajuster
       ensemble la position de la camera et les reglages de chaque vue.
    Renvoie (C, infos) ; C = None si le terrain n'a pas pu etre suivi."""
    pr = prior or CAMERA_PRIOR
    C0 = np.array([L * pr[0], W + pr[1], pr[2]])
    model = PitchModel(L, W)
    tr = PitchTracker(C0, L, W)
    sel = []
    t0 = time.time()
    for i, fr in enumerate(frames):
        img = to_work(fr)
        if i % 3:
            tr.motion(img)
            continue
        H, s = tr.full(img)
        if H is not None and s is not None and s < 1.0 and tr.p is not None:
            sel.append((img, tr.p.copy(), s))
    info = {"frames_seen": tr.stats["frames"], "frames_locked": len(sel),
            "track_s": round(time.time() - t0, 1), "prior": [round(float(c), 2) for c in C0]}
    if len(sel) < 8:
        info["error"] = "pitch lines not followed well enough (%d locked frames)" % len(sel)
        return None, info
    step = max(1, len(sel) // max_views)
    sel = sel[::step][:max_views]
    views = [View(im, tr.box) for im, _, _ in sel]
    P0 = np.array([p for _, p, _ in sel])
    cx, cy = tr._center()
    s0 = mean_score(views, model, C0, P0, cx, cy)
    t1 = time.time()
    # La hauteur est mal contrainte par les lignes seules (une camera plus haute et
    # plus loin, moins inclinee, dessine presque le meme terrain) : plusieurs
    # departs en hauteur, on garde l'ajustement qui colle le mieux.
    best = None
    for z0 in CAMERA_HEIGHTS:
        Cz = np.array([C0[0], C0[1], z0])
        Pz = np.array([local_search(v, model, Cz, P0[k], dp=(2.0, 1.0, 0.08), cx=cx, cy=cy)[0]
                       for k, v in enumerate(views)])
        x0 = np.concatenate([Cz, (Pz / np.array([1.0, 1.0, 1000.0])).ravel()])
        r = least_squares(_joint_residuals, x0, args=(views, model, cx, cy), method='trf',
                          loss='soft_l1', f_scale=0.3, max_nfev=40, x_scale='jac')
        if best is None or r.cost < best[0]:
            best = (r.cost, r.x)
    C = best[1][:3]
    P = best[1][3:].reshape(-1, 3) * np.array([1.0, 1.0, 1000.0])
    s1 = mean_score(views, model, C, P, cx, cy)
    info.update(views=len(views), fit_s=round(time.time() - t1, 1),
                score_prior=round(s0, 3), score=round(s1, 3), camera=[round(float(c), 2) for c in C])
    log("camera estimate: %s score %.3f (prior %.3f) from %d views" % (info["camera"], s1, s0, len(views)))
    if s1 > CAMERA_MAX_SCORE or not (0 < C[0] < L and W < C[1] < W + 40 and 1.0 < C[2] < 25.0):
        info["error"] = "camera fit not convincing"
        return None, info
    return C, info
