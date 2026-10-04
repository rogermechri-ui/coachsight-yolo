import cv2, numpy as np
from scipy.ndimage import median_filter

def field_mask(img, gap=14):
    """Pelouse : pour chaque colonne, on remonte depuis le bas tant que c'est
    de l'herbe (trous de moins de `gap` px toleres : lignes, joueurs), puis on
    lisse la limite haute (les arbres au-dessus des panneaux ne comptent pas)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (18, 40, 40), (95, 255, 255)) > 0
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)) > 0
    top = np.full(w, h, int)
    valid = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[h - 60:h - 20].mean(0) > 25     # colonnes avec de l'herbe en bas (pas les bandes noires)
    for x in np.where(valid)[0]:
        col = g[:, x]; y = h - 1; miss = 0; last = h - 1
        while y >= 0:
            if col[y]: last = y; miss = 0
            else:
                miss += 1
                if miss > gap: break
            y -= 1
        top[x] = last
    xs = np.where(valid)[0]
    if len(xs):
        sm = median_filter(top[xs], size=61, mode='nearest')
        top[xs] = sm.astype(int)
    f = np.zeros((h, w), np.uint8)
    for x in xs:
        f[top[x]:, x] = 255
    return f

def line_mask2(img):
    field = field_mask(img)
    h = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ridge = g8.astype(np.float32) - cv2.medianBlur(g8, 11).astype(np.float32)
    ridge2 = g8.astype(np.float32) - cv2.medianBlur(g8, 31).astype(np.float32)
    rows = np.arange(h)[:, None] / h
    fm = field > 0
    mad = float(np.median(np.abs(ridge[fm]))) if fm.any() else 3.0
    thr = max(5.0, 4.5 * mad)
    white = (((ridge > thr) | ((ridge2 > 18) & (rows > 0.6))) & (hsv[:, :, 1] < 120)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, cv2.erode(field, np.ones((5, 5), np.uint8)))
    # epaisseur toleree croissante vers le bas (perspective)
    dt = cv2.distanceTransform(white, cv2.DIST_L2, 3)
    lim = 2.5 + 6.0 * rows
    thick = (dt > lim).astype(np.uint8)
    if thick.any():
        white[cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, 8)
    span = np.maximum(st[1:, 2], st[1:, 3]); keep = np.zeros(n, bool)
    keep[1:] = (span >= 25) & (st[1:, 4] < span * 10)
    return (keep[lab] * 255).astype(np.uint8), field
