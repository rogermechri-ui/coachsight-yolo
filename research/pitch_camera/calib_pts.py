import sys,os; E=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,E)
import numpy as np
from cam import homog
from scipy.optimize import least_squares
cx, cy = 640.0, 268.0
def proj(H, X, Y):
    q = H @ [X, Y, 1.0]; return q[:2] / q[2]
def line_dist(H, pts_world_line, uv):
    # distance image du point uv a la droite image de la ligne terrain (2 points terrain)
    a = proj(H, *pts_world_line[0]); b = proj(H, *pts_world_line[1])
    d = b - a; n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return float((np.asarray(uv) - a) @ n)
obsE = {'L': 344.0, 'R': 927.6, 'T': 284.6, 'B': 339.4}
halfway = np.array([3.94e-02, 624.33])
def res(p, Lp=105.0, Wp=68.0):
    C = p[:3]
    r = []
    # image 464 (rond central)
    H = homog(C, p[3], p[4], p[5], cx, cy)
    t = np.linspace(0, 2 * np.pi, 360)
    P = np.array([proj(H, Lp / 2 + 9.15 * np.cos(a), Wp / 2 + 9.15 * np.sin(a)) for a in t])
    r += [P[:, 0].min() - obsE['L'], P[:, 0].max() - obsE['R'], P[:, 1].min() - obsE['T'], P[:, 1].max() - obsE['B']]
    for yy in (360, 450, 530):
        r.append(line_dist(H, [(Lp / 2, 0), (Lp / 2, Wp)], (np.polyval(halfway, yy), yy)))
    r.append(line_dist(H, [(0, 0), (Lp, 0)], (640, 263)))
    # image 100 (but de gauche)
    H = homog(C, p[6], p[7], p[8], cx, cy)
    r += list(proj(H, 0, Wp / 2 + 3.66) - [931, 259.5]); r += list(proj(H, 0, Wp / 2 - 3.66) - [1054, 256])
    for uv in [(611, 535), (300, 274), (455, 405)]:
        r.append(line_dist(H, [(0, Wp), (Lp, Wp)], uv))
    # image 250 (but de droite)
    H = homog(C, p[9], p[10], p[11], cx, cy)
    r += list(proj(H, Lp, Wp / 2 - 3.66) - [709.5, 212]); r += list(proj(H, Lp, Wp / 2 + 3.66) - [768.75, 214])
    for uv in [(770, 477), (1098, 280), (934, 378)]:
        r.append(line_dist(H, [(0, Wp), (Lp, Wp)], uv))
    return np.array(r)
best = None
for Yc in (70, 75, 80, 90):
    for Zc in (3, 5, 8):
        x0 = [52.5, Yc, Zc, 0, np.radians(4), 1100, np.radians(-70), np.radians(5), 900, np.radians(70), np.radians(5), 900]
        try:
            s = least_squares(res, x0, bounds=([40, 60, 0.5, -1, -0.5, 200, -2.5, -0.5, 100, 0, -0.5, 100], [65, 150, 30, 1, 1, 8000, 0, 1, 8000, 2.5, 1, 8000]))
        except Exception as e:
            continue
        if best is None or s.cost < best.cost: best = s
p = best.x
print('C', p[:3].round(2), 'rms px', np.sqrt(np.mean(best.fun ** 2)).round(2))
for k, o in (('464', 3), ('100', 6), ('250', 9)):
    print(k, 'pan %.1f tilt %.2f f %.0f' % (np.degrees(p[o]), np.degrees(p[o + 1]), p[o + 2]))
print('residuals', best.fun.round(1))
np.save(E + '/calib_pts.npy', p)
