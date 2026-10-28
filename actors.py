"""Tache `actor_crops` : images des joueurs autour de chaque action, pour que
l'application demande ensuite au modele de vision QUI a fait l'action
(numero de maillot parmi l'effectif). Le moteur ne lit pas les numeros : il
prepare des vues nettes et bien cadrees, c'est l'application qui interroge
le modele avec la liste fermee des numeros.

Entree (input) :
  video_url            lien de la video du match
  events               [{"id": str, "t": secondes, "team": "ours" | "theirs"}]
  team_colors          {"ours": "#2563eb", "theirs": "#f8fafc"}  (couleurs des maillots)
  keeper_colors        {"ours": "#e11d48", "theirs": "..."}  (facultatif)
  offsets_s            instants lus autour de t, defaut [-1, 0, 1]
  max_players          joueurs gardes par image, defaut 4
  crop_height          hauteur des vignettes (px), defaut 320
  context_width        largeur de l'image de contexte, defaut 1280
  upload_url           adresse signee ou deposer le zip (PUT)
Sortie (result) : {"events": [{"id", "t", "frames": [{"t", "context", "players": [
  {"label": "A", "file", "box": [x1,y1,x2,y2] (0..1 de l'image entiere),
   "height_px", "team_distance", "near_ball": bool, "ball_distance_px"}]}]}],
  "zip": {...}, "timings": {...}}
Le zip contient context_<event>_<t>.jpg (image reduite, joueurs candidats
entoures et lettres A, B, C...) et crop_<event>_<t>_<lettre>.jpg (vignettes).
"""
import io
import json
import os
import time
import zipfile

import cv2
import numpy as np
import requests

# Au-dela de cet ecart (0..1) a la couleur de l'equipe, la personne n'est pas
# consideree comme un joueur de cette equipe (arbitre, staff, chasuble, public).
TEAM_MAX_DIST = float(os.environ.get("ACTOR_TEAM_MAX_DIST", "0.45"))

# Part maximale de pixels du torse qui ressemblent a l'AUTRE equipe. Une boite qui
# recouvre un joueur bleu ET son voisin blanc donnait un percentile de clarte
# "blanc" : un joueur bleu etait propose pour une action des blancs (essai
# Brevard / Centre du 27 oct. 2026). Au-dela de ce seuil, la boite est ambigue :
# on l'ecarte plutot que de proposer un joueur de l'adversaire.
OPPOSITE_MAX = float(os.environ.get("ACTOR_OPPOSITE_MAX", "0.2"))
GRASS_HUE_MIN = int(os.environ.get("GRASS_HUE_MIN", "34"))
# Un candidat doit etre a moins de N fois sa hauteur du ballon.
NEAR_BALL_FACTOR = float(os.environ.get("ACTOR_NEAR_BALL_FACTOR", "2.5"))


def _hex_to_lab(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    px = np.uint8([[[b, g, r]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)


def _color_distance(col, ref):
    """Ecart entre une couleur mesuree (L, a, b[, rel]) et une couleur de reference
    Lab, insensible a l'intensite de la lumiere : on compare la direction de la
    teinte (angle dans le plan a, b) quand la reference est coloree, et la
    faiblesse de la teinte + la clarte quand la reference est blanche, grise ou
    noire. Les maillots filmes au soleil sont bien plus pales que la couleur
    saisie sur le site : une distance brute Lab classait des maillots bleus
    comme blancs."""
    L, a, b = [float(v) for v in col[:3]]
    Lr, ar, br = [float(v) for v in ref[:3]]
    ca, cb = a - 128.0, b - 128.0
    ra, rb = ar - 128.0, br - 128.0
    c_m, c_r = np.hypot(ca, cb), np.hypot(ra, rb)
    if c_r >= 20.0:                     # reference coloree : la teinte decide
        if c_m < 6.0:
            return 1.0                  # mesure sans teinte (blanc/gris) : loin d'une couleur
        cos = (ca * ra + cb * rb) / (c_m * c_r)
        ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return ang / 180.0 + 0.1 * abs(L - Lr) / 255.0
    # reference achromatique : peu de teinte et clarte proche
    return min(1.0, c_m / 40.0) + 0.6 * abs(L - Lr) / 255.0


def _classify(col, refs):
    """refs = {"ours": lab, "theirs": lab, ...}. Renvoie (nom, ecart 0..1) du plus proche."""
    best = None
    for name, ref in refs.items():
        d = _color_distance(col, ref)
        if best is None or d < best[1]:
            best = (name, d)
    return best


def _opposite_share(frame, box, refs, side):
    """Part (0..1) des pixels du torse (hors pelouse) plus proches de l'equipe adverse."""
    other = "theirs" if side == "ours" else "ours"
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    cx1, cx2 = x1 + int(w * 0.25), x2 - int(w * 0.25)
    crop = frame[max(y1 + int(h * 0.18), 0):y1 + int(h * 0.48), max(cx1, 0):max(cx2, cx1 + 1)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= GRASS_HUE_MIN) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50) & (hsv[:, 2] > 30)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)[~grass]
    if len(lab) < 12:
        return 0.0
    if len(lab) > 200:
        lab = lab[np.linspace(0, len(lab) - 1, 200).astype(int)]
    n = 0
    for c in lab:
        name, d = _classify(c, refs)
        if name == other and d <= TEAM_MAX_DIST:
            n += 1
    return n / len(lab)


def _draw_context(frame, picks, width):
    h, w = frame.shape[:2]
    s = width / w if w > width else 1.0
    img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s != 1.0 else frame.copy()
    for p in picks:
        x1, y1, x2, y2 = [int(v * s) for v in p["box_px"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, p["label"], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.putText(img, p["label"], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    return img


def actor_crops(inp, model, shirt_color, download, imgsz=1280):
    job_clock = time.time()
    events = inp.get("events") or []
    if not events:
        raise RuntimeError("no event")
    if len(events) > 300:
        raise RuntimeError("too many events: %d (max 300)" % len(events))
    offsets = [float(o) for o in (inp.get("offsets_s") or [-1, 0, 1])]
    max_players = int(inp.get("max_players", 4))
    crop_h = int(inp.get("crop_height", 320))
    ctx_w = int(inp.get("context_width", 1280))
    colors = inp.get("team_colors") or {}
    if not colors.get("ours") or not colors.get("theirs"):
        raise RuntimeError("team_colors.ours and team_colors.theirs are required")
    refs = {"ours": _hex_to_lab(colors["ours"]), "theirs": _hex_to_lab(colors["theirs"])}
    for side, hx in (inp.get("keeper_colors") or {}).items():
        if hx:
            refs["keeper_" + side] = _hex_to_lab(hx)

    clock = time.time()
    src = download(inp["video_url"])
    timings = {"download_s": round(time.time() - clock, 1)}
    cap = cv2.VideoCapture(src)
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED)
    out_events = []
    n_crops = 0
    t_read = t_detect = 0.0
    try:
        for ev in events:
            eid = str(ev["id"])
            t0 = float(ev["t"])
            side = "ours" if ev.get("team", "ours") == "ours" else "theirs"
            frames_out = []
            for off in offsets:
                t = max(0.0, t0 + off)
                c = time.time()
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                t_read += time.time() - c
                if not ok:
                    continue
                h, w = frame.shape[:2]
                c = time.time()
                res = model.predict(frame, classes=[0, 32], conf=0.25, imgsz=imgsz, verbose=False)[0]
                t_detect += time.time() - c
                persons, ball = [], None
                boxes = list(zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())) if res.boxes is not None else []
                balls = [(b, cf) for b, c, cf in boxes if int(c) == 32]
                if balls:
                    b = max(balls, key=lambda x: x[1])[0]      # le ballon le plus sur
                    ball = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if boxes:
                    for box, cls, _ in boxes:
                        if int(cls) == 32:
                            continue
                        col = shirt_color(frame, box)
                        if col is None:
                            continue
                        name, dist = _classify(col, refs)
                        if name != side or dist > TEAM_MAX_DIST:
                            continue          # autre equipe, arbitre, staff, chasuble
                        if _opposite_share(frame, box, refs, side) > OPPOSITE_MAX:
                            continue          # boite melangee (joueur adverse a cote) : ambigu
                        bh = box[3] - box[1]
                        if bh < 24:
                            continue          # trop petit : un numero y est illisible
                        feet = ((box[0] + box[2]) / 2, box[3])
                        bd = float(np.hypot(feet[0] - ball[0], feet[1] - ball[1])) if ball else None
                        persons.append({"box_px": box, "height_px": round(bh), "team_distance": round(dist, 3),
                                        "ball_distance_px": round(bd) if bd is not None else None})
                # Les plus proches du ballon d'abord ; sans ballon, les plus grands (= les plus proches de la camera).
                if ball:
                    persons.sort(key=lambda p: p["ball_distance_px"])
                else:
                    persons.sort(key=lambda p: -p["height_px"])
                # Sans ballon vu, ou trop loin du ballon (spectateur, remplacant, staff),
                # aucun candidat n'est propose : mieux vaut laisser vide que se tromper.
                if not ball:
                    persons = []
                else:
                    persons = [p for p in persons
                               if p["ball_distance_px"] is not None and p["ball_distance_px"] < NEAR_BALL_FACTOR * p["height_px"]]
                picks = persons[:max_players]
                for k, p in enumerate(picks):
                    p["label"] = chr(ord("A") + k)
                    p["near_ball"] = bool(ball) and p["ball_distance_px"] is not None and p["ball_distance_px"] < 2.5 * p["height_px"]
                tms = int(round(t * 1000))
                ctx_name = "context_%s_%09d.jpg" % (eid, tms)
                ok2, jpg = cv2.imencode(".jpg", _draw_context(frame, picks, ctx_w), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    zf.writestr(ctx_name, jpg.tobytes())
                players = []
                for p in picks:
                    x1, y1, x2, y2 = p["box_px"]
                    bw, bh = x2 - x1, y2 - y1
                    mx, my = 0.35 * bw, 0.15 * bh
                    X1, Y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
                    X2, Y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
                    crop = frame[Y1:Y2, X1:X2]
                    if crop.size == 0:
                        continue
                    s = crop_h / crop.shape[0]
                    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), crop_h),
                                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
                    name = "crop_%s_%09d_%s.jpg" % (eid, tms, p["label"])
                    ok3, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok3:
                        continue
                    zf.writestr(name, jpg.tobytes())
                    n_crops += 1
                    players.append({"label": p["label"], "file": name,
                                    "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                                    "height_px": p["height_px"], "team_distance": p["team_distance"],
                                    "near_ball": p["near_ball"], "ball_distance_px": p["ball_distance_px"]})
                frames_out.append({"t": round(t, 2), "context": ctx_name if ok2 else None,
                                   "ball_seen": bool(ball), "players": players})
            out_events.append({"id": eid, "t": t0, "team": side, "frames": frames_out})
        manifest = {"events": out_events, "team_colors": colors, "offsets_s": offsets}
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.close()
        data = buf.getvalue()
        zinfo = {"bytes": len(data), "crops": n_crops, "uploaded": False}
        if inp.get("upload_url"):
            clock = time.time()
            r = requests.put(inp["upload_url"], data=data, timeout=900,
                             headers={"Content-Type": "application/zip", "x-upsert": "true"})
            r.raise_for_status()
            zinfo["uploaded"] = True
            timings["upload_s"] = round(time.time() - clock, 1)
        if inp.get("keep_local"):            # tests locaux uniquement
            path = os.path.join(os.path.dirname(src) if os.path.isabs(src) else ".", "actor_crops.zip")
            with open(path, "wb") as f:
                f.write(data)
            zinfo["path"] = path
    finally:
        cap.release()
        if not inp.get("keep_local") and src != inp["video_url"] and os.path.exists(src):
            os.remove(src)
    timings.update(read_s=round(t_read, 1), detect_s=round(t_detect, 1), total_s=round(time.time() - job_clock, 1))
    return {"task": "actor_crops", "events": out_events, "zip": zinfo, "timings": timings}
