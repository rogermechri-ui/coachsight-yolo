import sys, time, pickle, os; E = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, E)
import cv2, numpy as np
from cam import *
from track import score, step
from scipy.optimize import least_squares

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def feat_mask(im):
    h, w = im.shape[:2]
    m = np.zeros((h, w), np.uint8); m[:, 175:1105] = 255
    m[h - 70:, 1117 - 140:] = 0
    return m

def rel_homography(a, b):
    """Mouvement de la camera entre deux images (homographie image a -> image b),
    a partir des details fixes (arbres, tribunes, panneaux)."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = orb.detectAndCompute(ga, feat_mask(a)); kb, db = orb.detectAndCompute(gb, feat_mask(b))
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, 0
    m = bf.match(da, db)
    if len(m) < 20:
        return None, 0
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    return H, int(inl.sum()) if inl is not None else 0

GRID = np.float32([[x, y] for x in np.linspace(200, 1080, 9) for y in np.linspace(280, 520, 5)])

def fit_params(C, Hpred, p0, cx, cy):
    """Cap / inclinaison / focale qui reproduisent le mieux l'homographie predite."""
    Hi = np.linalg.inv(Hpred)
    P = np.column_stack([GRID, np.ones(len(GRID))]) @ Hi.T
    ok = P[:, 2] > 0
    if ok.sum() < 6:
        return p0
    X = P[ok, :2] / P[ok, 2:3]; U = GRID[ok]
    def res(x):
        H = homog(C, x[0], x[1], x[2] * 1000, cx, cy)
        q = np.column_stack([X, np.ones(len(X))]) @ H.T
        return ((q[:, :2] / q[:, 2:3]) - U).ravel()
    r = least_squares(res, [p0[0], p0[1], p0[2] / 1000])
    return (r.x[0], r.x[1], r.x[2] * 1000)

if __name__ == '__main__':
    clip, start_idx, C, p0, tag = sys.argv[1], int(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), sys.argv[5]
    cap = cv2.VideoCapture(clip); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(im)
    res = {}
    for direction in (1, -1):
        p = p0; i = start_idx; prev = None; t = time.time()
        while 0 <= i < len(frames):
            im = frames[i]; fr = Frame(im)
            inl = -1
            if prev is not None:
                Hr, inl = rel_homography(prev, im)
                if Hr is not None and inl >= 30:
                    Hp = Hr @ homog(C, p[0], p[1], p[2], fr.cx, fr.cy)
                    p = fit_params(C, Hp, p, fr.cx, fr.cy)
            # ajustement fin sur les lignes, petite fenetre ; on ne l'accepte que s'il ameliore nettement
            s0 = float(score(fr, homog(C, *p[:2], p[2], fr.cx, fr.cy)[None])[0])
            q, s1 = step(fr, C, p, dp=(1.0, 0.4, 0.04))
            if s1 < s0 - 0.02 and s1 < 0.85:
                p, s = q, s1
            else:
                s = s0
            res[i] = (p, s, inl)
            prev = im; i += direction
        print('dir', direction, round(time.time() - t, 1), 's', flush=True)
    pickle.dump(res, open('%s/track_%s.pkl' % (E, tag), 'wb'))
