import cv2, numpy as np, sys
sys.path.insert(0, '/home/claude/coachsight-yolo')
import pitch_fit as pf
from mask2 import line_mask2

L, W = 105.0, 68.0

def homog(C, th, ph, f, cx, cy):
    """Homographie terrain (X,Y,0) -> image pour une camera en C, cap th, inclinaison ph, focale f."""
    fw = np.array([np.sin(th) * np.cos(ph), -np.cos(th) * np.cos(ph), -np.sin(ph)])
    rt = np.array([np.cos(th), np.sin(th), 0.0])
    dn = np.cross(rt, fw)
    R = np.stack([rt, dn, fw])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    t = -R @ np.asarray(C, float)
    return K @ np.column_stack([R[:, 0], R[:, 1], t])

class Frame:
    def __init__(self, img):
        self.img = img
        h, w = img.shape[:2]
        self.h, self.w = h, w
        white, field = line_mask2(img)
        x0, y0, x1, y1 = pf.content_box(img)
        white[h - 70:, x1 - 130:] = 0      # logo veo
        white[h - 70:, :x0 + 40] = 0       # icone
        self.white, self.field = white, field
        self.box = (x0, y0, x1, y1)
        self.cx, self.cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.dt = cv2.distanceTransform(255 - white, cv2.DIST_L2, 5)
        self.wpts = np.column_stack(np.where(white > 0))[:, ::-1].astype(np.float32)

SAMPLES = pf.sample_pitch(1.0, L, W).astype(np.float64)
SAMPLES_H = np.column_stack([SAMPLES, np.ones(len(SAMPLES))])

def cost_batch(fr, Hs, T=15.0, min_n=40):
    """Hs : (N,3,3). Cout terrain->lignes pour N candidats."""
    P = np.einsum('nij,mj->nmi', Hs, SAMPLES_H)
    z = P[:, :, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = P[:, :, 0] / z; v = P[:, :, 1] / z
    x0, y0, x1, y1 = fr.box
    ok = (z > 0) & (u >= x0) & (u < x1 - 1) & (v >= 0) & (v < fr.h - 1)
    ui = np.clip(u, 0, fr.w - 1).astype(int); vi = np.clip(v, 0, fr.h - 1).astype(int)
    onf = fr.field[vi, ui] > 0
    d = np.where(onf, np.minimum(fr.dt[vi, ui], T), T)   # trait dessine hors pelouse (arbres, ciel) = penalite maximale
    n = ok.sum(1)
    c = np.where((ok & onf).sum(1) >= min_n, (d * ok).sum(1) / np.maximum(n, 1), T)
    return c, n

def cost_bi(fr, H, T=15.0):
    """Cout dans les deux sens : dessin -> lignes et lignes detectees -> dessin."""
    c1, n = cost_batch(fr, H[None], T)
    canvas = np.zeros((fr.h, fr.w), np.uint8)
    for pl in pf.pitch_polylines(L, W):
        pts = np.column_stack([np.asarray(pl, float), np.ones(len(pl))]) @ H.T
        if (pts[:, 2] <= 0).any():
            # decoupe grossiere : on ne dessine que les segments devant la camera
            pass
        good = pts[:, 2] > 1e-6
        p = pts[:, :2] / np.where(good, pts[:, 2], 1)[:, None]
        for a, b, ga, gb in zip(p[:-1], p[1:], good[:-1], good[1:]):
            if ga and gb and np.all(np.abs(np.r_[a, b]) < 1e5):
                cv2.line(canvas, tuple(np.int32(a)), tuple(np.int32(b)), 255, 1)
    dm = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 5)
    if len(fr.wpts) == 0:
        return float(c1[0]), T
    wi = fr.wpts.astype(int)
    c2 = float(np.minimum(dm[wi[:, 1], wi[:, 0]], T).mean())
    return float(c1[0]), c2

def grid_search(fr, C, ths, phs, fs, keep=20, T=40.0):
    cands = []
    for f in fs:
        TH, PH = np.meshgrid(ths, phs, indexing='ij')
        Hs = np.stack([homog(C, a, b, f, fr.cx, fr.cy) for a, b in zip(TH.ravel(), PH.ravel())])
        c, n = cost_batch(fr, Hs, T=T)
        for i in np.argsort(c)[:keep]:
            cands.append((c[i], TH.ravel()[i], PH.ravel()[i], f))
    cands.sort(key=lambda x: x[0])
    return cands[:keep * 3]

from scipy.optimize import minimize

def refine(fr, C, th, ph, f):
    def obj(p):
        H = homog(C, p[0], p[1], p[2] * 1000, fr.cx, fr.cy)
        c1, c2 = cost_bi(fr, H)
        return c1 + c2
    r = minimize(obj, [th, ph, f / 1000], method='Powell',
                 options={'xtol': 1e-4, 'ftol': 1e-3, 'maxfev': 400})
    th, ph, f = r.x[0], r.x[1], r.x[2] * 1000
    return th, ph, f, r.fun

def solve(fr, C, coarse=True):
    ths = np.radians(np.arange(-75, 75.1, 1.0))
    phs = np.radians(np.arange(0, 20.1, 0.5))
    fs = np.geomspace(350, 5000, 30)
    cands = grid_search(fr, C, ths, phs, fs, keep=12)
    # reclassement des candidats avec le cout dans les deux sens
    scored = []
    for c, th, ph, f in cands:
        scored.append((sum(cost_bi(fr, homog(C, th, ph, f, fr.cx, fr.cy))), th, ph, f))
    scored.sort(key=lambda x: x[0])
    best = None
    for c, th, ph, f in scored[:6]:
        th2, ph2, f2, val = refine(fr, C, th, ph, f)
        if best is None or val < best[3]:
            best = (th2, ph2, f2, val)
    return best

# --- Distance (en metres) a la ligne de terrain la plus proche, sur une grille du terrain
RES = 0.1
PX0, PY0 = -25.0, -25.0
_gw, _gh = int((L + 50) / RES), int((W + 50) / RES)
_canvas = np.zeros((_gh, _gw), np.uint8)
for _pl in pf.pitch_polylines(L, W):
    _p = ((np.asarray(_pl) - [PX0, PY0]) / RES).astype(np.int32)
    cv2.polylines(_canvas, [_p], False, 255, 1)
PITCH_DT = cv2.distanceTransform(255 - _canvas, cv2.DIST_L2, 5) * RES

def white_sub(fr, n=300, seed=0):
    if not hasattr(fr, '_ws'):
        rng = np.random.default_rng(seed)
        w = fr.wpts
        fr._ws = w[rng.choice(len(w), min(n, len(w)), replace=False)] if len(w) else w
    return fr._ws

def cost2_batch(fr, Hs, Tm=1.5):
    """Lignes detectees -> dessin : chaque pixel blanc est ramene sur le terrain ;
    distance (m) a la ligne la plus proche."""
    w = white_sub(fr)
    if len(w) == 0:
        return np.full(len(Hs), Tm)
    Hi = np.linalg.inv(Hs)
    wh = np.column_stack([w, np.ones(len(w))]).astype(np.float64)
    P = np.einsum('nij,mj->nmi', Hi, wh)
    z = P[:, :, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        X = P[:, :, 0] / z; Y = P[:, :, 1] / z
    gx = ((X - PX0) / RES); gy = ((Y - PY0) / RES)
    ok = (z > 0) & (gx >= 0) & (gx < _gw - 1) & (gy >= 0) & (gy < _gh - 1)
    d = np.where(ok, PITCH_DT[np.clip(gy, 0, _gh - 1).astype(int), np.clip(gx, 0, _gw - 1).astype(int)], Tm)
    return np.minimum(d, Tm).mean(1)

def grid_search2(fr, C, ths, phs, fs, keep=40):
    cands = []
    TH, PH = np.meshgrid(ths, phs, indexing='ij'); TH = TH.ravel(); PH = PH.ravel()
    for f in fs:
        Hs = np.stack([homog(C, a, b, f, fr.cx, fr.cy) for a, b in zip(TH, PH)])
        c1, n = cost_batch(fr, Hs, T=40.0)
        c2 = cost2_batch(fr, Hs)
        s = c1 / 40.0 + c2 / 1.5
        for i in np.argsort(s)[:keep]:
            cands.append((s[i], TH[i], PH[i], f))
    cands.sort(key=lambda x: x[0])
    return cands

def solve2(fr, C, nref=6):
    ths = np.radians(np.arange(-75, 75.1, 1.0))
    phs = np.radians(np.arange(0, 20.1, 0.5))
    fs = np.geomspace(350, 5000, 30)
    cands = grid_search2(fr, C, ths, phs, fs, keep=10)[:60]
    scored = sorted(((sum(cost_bi(fr, homog(C, th, ph, f, fr.cx, fr.cy))), th, ph, f) for _, th, ph, f in cands), key=lambda x: x[0])
    best = None
    for c, th, ph, f in scored[:nref]:
        r = refine(fr, C, th, ph, f)
        if best is None or r[3] < best[3]:
            best = r
    return best
