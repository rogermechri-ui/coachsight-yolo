import sys, time, pickle, os; E = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, E)
import cv2, numpy as np
from cam import *
from scipy.optimize import minimize

def score(fr, Hs):
    c1, n = cost_batch(fr, Hs, T=15.0)
    c2 = cost2_batch(fr, Hs)
    return c1 / 15.0 + c2 / 1.5

def step(fr, C, p, dp=(2.5, 0.8, 0.10)):
    th, ph, f = p
    ths = th + np.radians(np.linspace(-dp[0], dp[0], 21))
    phs = ph + np.radians(np.linspace(-dp[1], dp[1], 9))
    fs = f * np.linspace(1 - dp[2], 1 + dp[2], 9)
    G = np.array(np.meshgrid(ths, phs, fs, indexing='ij')).reshape(3, -1).T
    Hs = np.stack([homog(C, a, b, c, fr.cx, fr.cy) for a, b, c in G])
    s = score(fr, Hs)
    i = int(np.argmin(s)); q = G[i]
    obj = lambda x: float(score(fr, homog(C, x[0], x[1], x[2] * 1000, fr.cx, fr.cy)[None])[0])
    r = minimize(obj, [q[0], q[1], q[2] / 1000], method='Powell', options={'xtol': 1e-4, 'ftol': 1e-4, 'maxfev': 200})
    return (r.x[0], r.x[1], r.x[2] * 1000), r.fun

if __name__ == '__main__':
    clip, start_idx, C, p0, tag = sys.argv[1], int(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), sys.argv[5]
    cap = cv2.VideoCapture(clip); frames = []
    while True:
        ok, im = cap.read()
        if not ok: break
        frames.append(im)
    print('frames', len(frames), frames[0].shape, flush=True)
    res = {}
    for direction in (1, -1):
        p = p0; i = start_idx
        t = time.time()
        while 0 <= i < len(frames):
            fr = Frame(frames[i])
            p, s = step(fr, C, p)
            res[i] = (p, s)
            i += direction
        print('dir', direction, 'done', round(time.time() - t, 1), 's', flush=True)
    pickle.dump(res, open('%s/track_%s.pkl' % (E, tag), 'wb'))
    sc = np.array([res[i][1] for i in sorted(res)])
    print('score quantiles', np.percentile(sc, [10, 50, 90, 99]).round(3))
