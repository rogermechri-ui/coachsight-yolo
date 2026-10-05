"""CoachSight AI - moteur de detection YOLO pour RunPod Serverless.
Recoit la video d'un match, detecte joueurs et ballon, lit les numeros de
maillot quand ils sont visibles et renvoie le tout a l'application
(callback_url)."""
import io
import json
import os
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import cv2
import numpy as np
import requests
import runpod
from ultralytics import YOLO

import pitch_fit
import pitch_track

try:
    import pytesseract
except Exception:  # OCR indisponible : le suivi continue sans numeros
    pytesseract = None

MODEL = YOLO(os.environ.get("YOLO_MODEL", "yolov8m.pt"))
PITCH_L, PITCH_W = 105.0, 68.0
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2"))
READ_NUMBERS = os.environ.get("READ_NUMBERS", "0") != "0"
# Lire les numeros une image sur N : suffisant pour un vote majoritaire
# fiable par piste, sans exploser la duree du traitement.
OCR_EVERY_N = int(os.environ.get("OCR_EVERY_N", "5"))
# Images envoyees a l'application pour construire stats et rapport cote
# serveur (le navigateur du coach n'a plus a relire la video).
FRAMES_EVERY_S = float(os.environ.get("FRAMES_EVERY_S", "12"))
FRAMES_WIDTH = int(os.environ.get("FRAMES_WIDTH", "960"))
FRAMES_QUALITY = int(os.environ.get("FRAMES_QUALITY", "70"))
# Nombre de lectures video en parallele pour extraire ces images.
FRAMES_WORKERS = int(os.environ.get("FRAMES_WORKERS", str(min(6, os.cpu_count() or 1))))
# Lecture continue de la video pendant le tracking (plus rapide que de
# "sauter" a chaque image). Mettre SEQUENTIAL_READ=0 pour revenir a l'ancien mode.
SEQUENTIAL_READ = os.environ.get("SEQUENTIAL_READ", "1") != "0"
# Lire la video directement depuis son lien (seul le segment demande est
# telecharge) au lieu de copier tout le fichier. STREAM_VIDEO=0 = ancien mode.
STREAM_VIDEO = os.environ.get("STREAM_VIDEO", "0") != "0"
# Reconnexion automatique si le flux video est coupe en cours de lecture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "reconnect;1|reconnect_streamed;1|reconnect_on_network_error;1|reconnect_delay_max;10")
# Calage du terrain image par image (camera Veo fixe, voir pitch_track.py).
# Position de la camera par defaut "x,y,z" en metres (sinon envoyee par
# l'application dans pitch.camera) ; vide = pas de calage automatique.
PITCH_CAMERA = os.environ.get("PITCH_CAMERA", "").strip()
# Mouvement de la camera suivi plusieurs fois par seconde (panoramiques rapides).
MOTION_FPS = float(os.environ.get("MOTION_FPS", "6"))
# Recalage fin sur les lignes une image analysee sur N (les autres : mouvement seul).
PITCH_FULL_EVERY = int(os.environ.get("PITCH_FULL_EVERY", "2"))
# Detection : on retire les bandes noires des videos Veo (image 16:9 dans un
# cadre 21:9) et on analyse a une taille suffisante pour voir les joueurs
# lointains (640 = taille par defaut de YOLO : la moitie des joueurs echappent).
CROP_BARS = os.environ.get("CROP_BARS", "1") != "0"
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "960"))
# Hauteur minimale (pixels) d'un joueur pour tenter de lire son numero.
# En dessous (cas des videos panoramiques), les chiffres sont illisibles et
# la lecture ne fait que ralentir l'analyse.
OCR_MIN_HEIGHT = int(os.environ.get("OCR_MIN_HEIGHT", "90"))


def shirt_color(frame, box):
    """Couleur du maillot : centre du torse, en ignorant la pelouse (verte)
    qui entoure les petits joueurs sur une video panoramique."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    cx1, cx2 = x1 + int(w * 0.25), x2 - int(w * 0.25)
    crop = frame[max(y1 + int(h * 0.18), 0):y1 + int(h * 0.48), max(cx1, 0):max(cx2, cx1 + 1)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50) & (hsv[:, 2] > 30)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = lab[~grass]
    if len(keep) < max(4, 0.15 * len(lab)):
        return None   # presque tout est de la pelouse : mesure non fiable
    return np.median(keep, axis=0)


def read_number(frame, box):
    """Tente de lire un numero de maillot (1 a 99) sur le torse ou le dos."""
    if pytesseract is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    # Trop petit pour etre lisible : on ne tente pas.
    if h < OCR_MIN_HEIGHT or w < 24:
        return None
    crop = frame[y1 + int(h * 0.10):y1 + int(h * 0.55), max(x1, 0):x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    for img in (binary, cv2.bitwise_not(binary)):
        try:
            txt = pytesseract.image_to_string(img, config=config).strip()
        except Exception:
            return None
        if txt.isdigit() and 1 <= int(txt) <= 99:
            return int(txt)
    return None


class FrameCollector:
    """Garde une petite image JPEG toutes les `every_s` secondes, puis les
    envoie en un seul zip (avec un index manifest.json) vers une URL signee."""

    def __init__(self, inp):
        self.url = inp.get("frames_upload_url")
        self.every = float(inp.get("frames_every_s", FRAMES_EVERY_S))
        self.width = int(inp.get("frames_width", FRAMES_WIDTH))
        self.quality = int(inp.get("frames_quality", FRAMES_QUALITY))
        self.buf = io.BytesIO()
        self.zip = zipfile.ZipFile(self.buf, "w", zipfile.ZIP_STORED)
        self.index = []
        self.last_t = None

    @property
    def enabled(self):
        return bool(self.url) and self.every > 0

    def maybe_add(self, t, frame):
        if not self.enabled or (self.last_t is not None and t - self.last_t < self.every - 1e-6):
            return
        self.last_t = t
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        self.add_jpeg(t, jpg.tobytes())

    def add_jpeg_frame(self, t, frame):
        """Ajoute l'image sans la regle d'espacement (instants deja choisis)."""
        self.last_t = None
        self.maybe_add(t, frame)

    def add_jpeg(self, t, data):
        self.last_t = t
        name = "frame_%09d.jpg" % int(round(t * 1000))
        self.zip.writestr(name, data)
        self.index.append({"t": round(t, 2), "file": name})

    def upload(self):
        """Envoie le zip. Une erreur ici ne fait jamais echouer le tracking."""
        if not self.enabled:
            return {"uploaded": False, "count": 0, "reason": "no frames_upload_url"}
        info = {"uploaded": False, "count": len(self.index), "every_s": self.every, "width": self.width}
        try:
            self.zip.writestr("manifest.json", json.dumps({"frames": self.index}))
            self.zip.close()
            data = self.buf.getvalue()
            info["bytes"] = len(data)
            r = requests.put(self.url, data=data, timeout=600,
                             headers={"Content-Type": "application/zip", "x-upsert": "true"})
            r.raise_for_status()
            info["uploaded"] = True
        except Exception as e:
            info["error"] = str(e)[:300]
        return info


def iter_frames(cap, start, end, fps_out):
    """Donne (t, image) toutes les 1/fps_out secondes entre start et end.
    Mode continu : on se place une seule fois au debut puis on lit la video
    dans l'ordre (grab sans decodage complet des images ignorees)."""
    step = 1.0 / fps_out
    vfps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not SEQUENTIAL_READ or not (1 <= vfps <= 240):
        t = start
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                return
            yield t, frame
            t += step
        return
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    next_t = start
    t0 = None
    n = 0
    while True:
        if not cap.grab():
            return
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t0 is None:
            t0 = pos if pos > 0 else start
        t = pos if pos > 0 else t0 + n / vfps
        n += 1
        if t > end + 1e-6:
            return
        if t + 0.5 / vfps < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            return
        yield next_t, frame
        next_t += step


def _grab_jpegs(source, times, width, quality):
    """Lit les images aux instants `times` (fichier local) et les compresse en
    JPEG. Chaque appel a sa propre lecture video : plusieurs appels peuvent
    tourner en parallele (OpenCV libere le verrou Python pendant le decodage)."""
    cap = cv2.VideoCapture(source)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out.append((t, jpg.tobytes()))
    finally:
        cap.release()
    return out


def collect_frames(cap, inp, windows, source=None):
    """Passe rapide : une image toutes les `frames_every_s` secondes dans chaque
    periode de jeu (`windows`), puis envoi. Avec un fichier local (`source`),
    la lecture est repartie sur plusieurs fils en parallele (FRAMES_WORKERS)."""
    collector = FrameCollector(inp)
    if not collector.enabled:
        return collector.upload()
    times = []
    for ws, we in windows:
        t = ws
        while t <= we:
            times.append(t)
            t += collector.every
    if source and FRAMES_WORKERS > 1 and times:
        n = min(FRAMES_WORKERS, len(times))
        size = -(-len(times) // n)   # parts contigues, une par fil
        parts = [times[i:i + size] for i in range(0, len(times), size)]
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            results = list(pool.map(lambda p: _grab_jpegs(source, p, collector.width, collector.quality), parts))
        for part, res in zip(parts, results):
            for t, jpg in res:
                collector.add_jpeg(t, jpg)
            if len(res) < len(part):
                break   # fin de video atteinte dans cette partie : on s'arrete la
        return collector.upload()
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        collector.add_jpeg_frame(t, frame)
    return collector.upload()


def download(url):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return tmp


def open_video(url, start):
    """Ouvre la video en lecture directe depuis son lien si possible (seul le
    segment utile transite), sinon la telecharge entierement (ancien mode).
    Renvoie (cap, fichier_temporaire_ou_None, mode, raison_si_echec)."""
    why = None   # lecture directe desactivee : rien a signaler
    if STREAM_VIDEO:
        clock = time.time()
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                ok, _ = cap.read()
                if ok:
                    return cap, None, "stream", None
                why = "opened but first read failed"
            else:
                why = "could not open link"
            cap.release()
        except Exception as e:
            why = "error: %s" % str(e)[:120]
        why += " (%.1f s)" % (time.time() - clock)
    tmp = download(url)
    return cv2.VideoCapture(tmp), tmp, "download", why


def video_end(cap):
    """Duree de la video en secondes (None si inconnue)."""
    n, vfps = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FPS) or 0
    return n / vfps if n > 0 and vfps > 0 else None


class PitchWorker:
    """Calage du terrain en tache de fond, pendant que le GPU detecte les joueurs.
    Recoit les images dans l'ordre ; pour chaque image analysee, garde
    l'homographie terrain -> image (ou None)."""

    def __init__(self, camera, L, W):
        self.tracker = pitch_track.PitchTracker(camera, L, W)
        self.q = queue.Queue(maxsize=32)
        self.H = {}          # indice de l'image analysee -> (H, score)
        self.size = None     # taille de l'image de travail (w, h)
        self.error = None
        self.n_full = 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def put(self, kind, frame=None, idx=None):
        # Image reduite tout de suite : file d'attente legere (2 Mo par image).
        self.q.put((kind, None if frame is None else pitch_track.to_work(frame), idx))

    def _run(self):
        while True:
            kind, frame, idx = self.q.get()
            if kind == "stop":
                return
            if self.error:
                continue
            try:
                if kind == "reset":            # nouvelle periode : la camera a pu bouger
                    self.tracker.p = None
                    self.tracker.feat = None
                    continue
                img = pitch_track.to_work(frame)
                self.size = (img.shape[1], img.shape[0])
                tr = self.tracker
                if kind == "motion":
                    tr.motion(img)
                    continue
                # image analysee : recalage sur les lignes une fois sur N, sinon mouvement seul
                if self.n_full % PITCH_FULL_EVERY == 0 or tr.p is None:
                    H, s = tr.full(img)
                else:
                    tr.motion(img)
                    H, s = tr.current(), None
                self.n_full += 1
                self.H[idx] = (H, s)
            except Exception as e:  # le calage ne doit jamais faire echouer l'analyse
                self.error = "%s: %s" % (type(e).__name__, str(e)[:200])

    def finish(self):
        self.put("stop")
        self.th.join()
        return self.H


def track(cap, windows, fps_out, timings, pitch=None):
    """Detection + suivi des joueurs et du ballon dans chaque periode de jeu.
    Les positions sont gardees en coordonnees d'image normalisees (0..1) :
    la conversion en metres se fait ensuite (calage du terrain si fourni).
    pitch : PitchWorker optionnel, alimente avec les images (analysees et
    intermediaires) pour caler le terrain image par image."""
    raw = []      # (t, [(trackId, u, v, ball, color)])
    votes = {}    # trackId -> numeros lus (vote majoritaire en fin de traitement)
    frame_i = 0
    t_read = t_detect = t_color = t_ocr = 0.0
    MODEL.predictor = None   # repart d'un suivi vierge (cas d'une 2e tentative)
    k = max(1, int(round(MOTION_FPS / fps_out))) if pitch else 1
    crop = None     # zone utile de l'image (sans les bandes noires), fixee a la 1re image
    for ws, we in windows:
        clock = time.time()
        if pitch:
            pitch.put("reset")
        for j, (t, frame) in enumerate(iter_frames(cap, ws, we, fps_out * k)):
            t_read += time.time() - clock
            if j % k:
                pitch.put("motion", frame)
                clock = time.time()
                continue
            if pitch:
                pitch.put("full", frame, len(raw))
            h, w = frame.shape[:2]
            c = time.time()
            if crop is None:
                crop = pitch_fit.content_box(frame) if CROP_BARS else (0, 0, w, h)
                if crop[2] - crop[0] < w // 2 or crop[3] - crop[1] < h // 2:
                    crop = None          # image noire (debut de video) : on reessaiera
            x0, y0, x1, y1 = crop or (0, 0, w, h)
            # Detection seule (pas le suivi integre de YOLO : a 2 images/s avec une
            # camera qui bouge, il ne garde que quelques joueurs et jette les autres).
            # Les identifiants sont attribues ensuite, sur le terrain (assign_ids).
            res = MODEL.predict(frame[y0:y1, x0:x1], classes=[0, 32], conf=0.25,
                                imgsz=YOLO_IMGSZ, verbose=False)[0]
            t_detect += time.time() - c
            pts = []
            if res.boxes is not None:
                ids = res.boxes.id.tolist() if res.boxes.id is not None else [None] * len(res.boxes)
                for box, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), ids):
                    box = [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0]
                    ball = int(cls) == 32
                    cx, cy = (box[0] + box[2]) / 2, box[3] if not ball else (box[1] + box[3]) / 2
                    c = time.time()
                    col = None if ball else shirt_color(frame, box)
                    t_color += time.time() - c
                    tid_i = int(tid) if tid is not None else -1
                    if (not ball and READ_NUMBERS and pytesseract is not None
                            and tid_i >= 0 and frame_i % OCR_EVERY_N == 0):
                        c = time.time()
                        n = read_number(frame, box)
                        t_ocr += time.time() - c
                        if n is not None:
                            votes.setdefault(tid_i, Counter()).update([n])
                    pts.append((tid_i, cx / w, cy / h, ball, col))
            raw.append((round(t, 2), pts))
            frame_i += 1
            clock = time.time()
    timings.update(read_s=round(t_read, 1), detect_s=round(t_detect, 1),
                   color_s=round(t_color, 1), ocr_s=round(t_ocr, 1),
                   numbers_read=sum(sum(c.values()) for c in votes.values()))
    return raw, votes


def play_windows(inp, start, end):
    """Periodes de jeu demandees (ex. [[30, 3430], [4430, 5400]]) limitees au
    segment ; sans periodes, tout le segment. La mi-temps n'est pas analysee."""
    out = []
    for w in inp.get("play_windows") or []:
        try:
            ws, we = max(float(w[0]), start), min(float(w[1]), end)
        except (TypeError, ValueError, IndexError):
            continue
        if we - ws >= 1:
            out.append([ws, we])
    out.sort()
    return out or [[start, end]]


def build_homography(calib):
    """Calage du terrain : au moins 4 reperes {"image": [u, v] (0..1),
    "pitch": [x, y] (metres)}. Renvoie (H, erreur moyenne en metres)."""
    pts = (calib or {}).get("points") or []
    if len(pts) < 4:
        return None, None
    img = np.float32([p["image"] for p in pts])
    pit = np.float32([p["pitch"] for p in pts])
    H, _ = cv2.findHomography(img, pit, 0)
    if H is None:
        return None, None
    proj = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, float(np.mean(np.linalg.norm(proj - pit, axis=1)))


def to_pitch(H, u, v, L, W):
    if H is None:   # sans calage : approximation lineaire (ancien comportement)
        return u * L, v * W
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def pitch_camera(inp):
    """Position de la camera (metres) : pitch.camera envoye par l'application
    ({x, y, z} ou [x, y, z]), sinon PITCH_CAMERA ; None = pas de calage auto."""
    cam = (inp.get("pitch") or {}).get("camera")
    try:
        if isinstance(cam, dict):
            return np.array([float(cam["x"]), float(cam["y"]), float(cam["z"])])
        if isinstance(cam, (list, tuple)) and len(cam) == 3:
            return np.array([float(c) for c in cam])
        if PITCH_CAMERA:
            return np.array([float(c) for c in PITCH_CAMERA.split(",")])
    except (KeyError, TypeError, ValueError):
        return None
    return None


# Memoire des joueurs sortis de l'image (camera Veo qui suit le ballon : on ne
# voit qu'une partie du terrain a chaque instant).
MEMORY_S = float(os.environ.get("PLAYER_MEMORY_S", "6"))
TACTICAL_MIN_PLAYERS = int(os.environ.get("TACTICAL_MIN_PLAYERS", "8"))


def assign_ids(placed, gate=4.0, max_gap_s=1.5):
    """Identifiants de joueurs d'une image a l'autre, par plus proche voisin sur
    le terrain (metres) : insensible aux mouvements de la camera. placed : liste
    de (t, [(tid, x, y, ball, col)]) ; renvoie la meme liste avec les tid."""
    from scipy.optimize import linear_sum_assignment
    tracks = {}      # id -> (t, x, y)
    next_id = 0
    out = []
    for t, pts in placed:
        players = [i for i, p in enumerate(pts) if not p[3]]
        live = [k for k, (lt, _, _) in tracks.items() if t - lt <= max_gap_s]
        ids = [None] * len(pts)
        if players and live:
            D = np.array([[np.hypot(pts[i][1] - tracks[k][1], pts[i][2] - tracks[k][2]) for k in live]
                          for i in players])
            r, c = linear_sum_assignment(D)
            for a, b in zip(r, c):
                if D[a, b] <= gate * max(1.0, (t - tracks[live[b]][0]) / 0.5):
                    ids[players[a]] = live[b]
        new = []
        for i, p in enumerate(pts):
            if p[3]:
                new.append((-1,) + tuple(p[1:]))
                continue
            if ids[i] is None:
                ids[i] = next_id
                next_id += 1
            tracks[ids[i]] = (t, p[1], p[2])
            new.append((ids[i],) + tuple(p[1:]))
        out.append((t, new))
    return out


def with_memory(frames, per_frame_H, size, memory_s=MEMORY_S):
    """Pour chaque image calee, ajoute aux joueurs visibles la derniere position
    connue (moins de memory_s secondes) de ceux qui sont sortis du champ de la
    camera. Un joueur memorise dont la position tombe dans la partie visible du
    terrain est oublie : s'il etait encore la, il serait detecte (ou il a change
    de numero de piste). Renvoie une nouvelle liste d'images (pour les mesures
    tactiques uniquement)."""
    out = []
    mem = {}     # trackId -> (t, x, y, team)
    w, h = size or (0, 0)
    for i, f in enumerate(frames):
        Hf = per_frame_H.get(i, (None, None))[0]
        if not f.get("calibrated") or Hf is None:
            out.append(f)
            continue
        t = f["t"]
        seen = set()
        pts = list(f["points"])
        for p in pts:
            if not p["ball"] and p["trackId"] >= 0 and p["team"] is not None:
                mem[p["trackId"]] = (t, p["x"], p["y"], p["team"])
                seen.add(p["trackId"])
        for tid in list(mem):
            mt, x, y, team = mem[tid]
            if t - mt > memory_s or t < mt:
                del mem[tid]
                continue
            if tid in seen:
                continue
            q = Hf @ np.array([x, y, 1.0])
            if q[2] > 0 and 0 <= q[0] / q[2] < w and 0 <= q[1] / q[2] < h:
                del mem[tid]          # devrait etre visible : position perimee
                continue
            pts.append({"trackId": tid, "team": team, "x": x, "y": y, "ball": False,
                        "number": None, "remembered": True})
        out.append({"t": t, "points": pts, "calibrated": True})
    return out


def tactical_metrics(frames, windows, L, W, min_players=6):
    """Indicateurs par periode et par equipe (groupes 0 et 1), a partir des
    positions en metres (necessite un terrain cale) :
    hauteur de la ligne defensive, largeur du bloc, longueur du bloc (compacite)
    et part du jeu dans chaque tiers. Le gardien (joueur le plus proche de son
    but) est exclu des mesures de bloc."""
    out = []
    for ws, we in windows:
        fr = [f for f in frames if ws <= f["t"] <= we]
        per_team = {0: [], 1: []}
        for f in fr:
            for team in (0, 1):
                xs = [(p["x"], p["y"]) for p in f["points"] if not p["ball"] and p["team"] == team]
                if len(xs) >= min_players:
                    per_team[team].append(xs)
        if not per_team[0] or not per_team[1]:
            out.append({"window": [ws, we], "frames_used": 0, "teams": {}})
            continue
        # Cote defendu : l'equipe dont le centre moyen est le plus a gauche defend x = 0.
        mean_x = {t: float(np.mean([np.mean([p[0] for p in xs]) for xs in per_team[t]])) for t in (0, 1)}
        left = 0 if mean_x[0] <= mean_x[1] else 1
        teams = {}
        for team in (0, 1):
            own_left = team == left
            lines, widths, lengths, thirds = [], [], [], [0, 0, 0]
            for xs in per_team[team]:
                d = sorted(((x if own_left else L - x), y) for x, y in xs)   # distance a son propre but
                outfield = d[1:]                                               # sans le gardien
                lines.append(float(np.mean([p[0] for p in outfield[:3]])))
                ys = [p[1] for p in outfield]
                widths.append(max(ys) - min(ys))
                lengths.append(outfield[-1][0] - outfield[0][0])
                c = float(np.mean([p[0] for p in outfield]))                    # centre du bloc
                thirds[min(2, max(0, int(c / (L / 3))))] += 1
            n = len(per_team[team])
            teams[str(team)] = {
                "frames": n,
                "defensive_line_m": round(float(np.median(lines)), 1),
                "block_width_m": round(float(np.median(widths)), 1),
                "block_length_m": round(float(np.median(lengths)), 1),
                "block_in_thirds_pct": {"defensive": round(100 * thirds[0] / n), "middle": round(100 * thirds[1] / n),
                                         "attacking": round(100 * thirds[2] / n)},
                "defends": "left" if own_left else "right",
            }
        out.append({"window": [ws, we], "frames_used": min(len(per_team[0]), len(per_team[1])), "teams": teams})
    return out


def process(inp):
    url, start, end = inp["video_url"], float(inp.get("start_s", 0)), float(inp["end_s"])
    fps_out = min(float(inp.get("fps", SAMPLE_FPS)), SAMPLE_FPS)
    pitch = inp.get("pitch") or {}
    L, W = float(pitch.get("length", PITCH_L)), float(pitch.get("width", PITCH_W))
    H, calib_err = build_homography(inp.get("pitch_calibration"))
    timings = {}
    clock = job_clock = time.time()
    cap, tmp, mode, why = open_video(url, start)
    timings["open_s"] = round(time.time() - clock, 1)
    timings["video_mode"] = mode
    if why:
        timings["stream_fallback"] = "direct read not used: " + why
    h0, w0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    timings["video_size"] = "%dx%d" % (w0, h0)
    dur = video_end(cap)
    if dur:
        end = min(end, dur)
    windows = play_windows(inp, start, end)
    timings["analyzed_s"] = round(sum(we - ws for ws, we in windows), 1)
    # Images pour stats/rapport : extraites et envoyees AVANT le tracking,
    # pour que le lien d'envoi (valable ~2 h) ne soit jamais expire.
    clock = time.time()
    frames_zip = collect_frames(cap, inp, windows, source=tmp)
    timings["frames_s"] = round(time.time() - clock, 1)
    if inp.get("early_frames_callback") and frames_zip.get("uploaded"):
        notify(inp, status="frames_ready", frames_zip=frames_zip, timings=dict(timings))
    camera = pitch_camera(inp)
    clock = time.time()
    worker = PitchWorker(camera, L, W) if camera is not None else None
    raw, votes = track(cap, windows, fps_out, timings, worker)
    expected = int(sum(we - ws for ws, we in windows) * fps_out) + len(windows)
    # Lecture directe interrompue (coupure reseau) : on recommence a partir
    # d'une copie complete de la video pour ne rien perdre.
    if mode == "stream" and len(raw) < 0.95 * expected:
        timings["stream_fallback"] = "%d/%d images lues" % (len(raw), expected)
        cap.release()
        tmp = download(url)
        cap = cv2.VideoCapture(tmp)
        timings["video_mode"] = "download"
        if worker:
            worker.finish()
            worker = PitchWorker(camera, L, W)
        raw, votes = track(cap, windows, fps_out, timings, worker)
    timings["tracking_s"] = round(time.time() - clock, 1)
    per_frame_H, pitch_info = {}, None
    if worker:
        c = time.time()
        per_frame_H = worker.finish()
        timings["pitch_wait_s"] = round(time.time() - c, 1)     # attente du calage apres le suivi
        st = worker.tracker.stats
        timings["pitch_s"] = round(st["time_s"], 1)
        n_cal = sum(1 for H, _ in per_frame_H.values() if H is not None)
        pitch_info = {"method": "camera_tracking", "camera": [round(float(x), 2) for x in camera],
                      "frames": len(raw), "frames_calibrated": n_cal,
                      "frames_calibrated_pct": round(100.0 * n_cal / max(1, len(raw)), 1),
                      "global_searches": st["global_searches"], "relocks": st["relocks"],
                      "motion_failed": st["motion_failed"], "error": worker.error}
    cap.release()
    if tmp:
        os.remove(tmp)

    # Positions en metres ; avec calage, tout ce qui est hors du terrain
    # (public, bancs, arbitres de touche) est ecarte.
    margin = 2.0
    placed = []
    cal_flags = []
    for i, (tt, pts) in enumerate(raw):
        Hf = per_frame_H.get(i, (None, None))[0]
        keep = []
        for tid, u, v, ball, col in pts:
            if Hf is not None:
                xy = pitch_track.image_to_pitch(Hf, u, v, *worker.size)
                if xy is None:
                    continue
                x, y = xy
            else:
                x, y = to_pitch(H, u, v, L, W)
            if (Hf is not None or H is not None) and not (-margin <= x <= L + margin and -margin <= y <= W + margin):
                continue
            keep.append((tid, x, y, ball, col))
        placed.append((tt, keep))
        cal_flags.append(Hf is not None or H is not None)

    placed = assign_ids(placed)
    cols = np.array([p[4] for _, pts in placed for p in pts if p[4] is not None], dtype=np.float32)
    centers = None
    if len(cols) >= 4:
        _, _, centers = cv2.kmeans(cols, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    # Numero retenu par piste : celui le plus souvent lu (vote majoritaire).
    numbers = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    tracks = []
    seen = {}
    frames = []
    for (tt, pts), cal in zip(placed, cal_flags):
        out = []
        for tid, x, y, ball, col in pts:
            team = None
            if col is not None and centers is not None:
                team = int(np.argmin(np.linalg.norm(centers - col, axis=1)))
            num = None if ball else numbers.get(tid)
            out.append({
                "trackId": tid, "team": team,
                "x": round(float(x), 2), "y": round(float(y), 2),
                "ball": ball,
                "number": num,
            })
            if not ball and tid >= 0 and tid not in seen:
                seen[tid] = True
                tracks.append({"track_id": tid, "team": team, "number": num})
        frames.append({"t": tt, "points": out, "calibrated": bool(cal)})
    # Couleur moyenne du maillot de chaque groupe (0 et 1), pour que
    # l'application relie chaque groupe a l'equipe du coach ou a l'adversaire.
    team_colors = []
    if centers is not None:
        for i, lab in enumerate(centers):
            px = np.uint8([[np.clip(lab, 0, 255)]])
            b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0][0]
            team_colors.append({"team": i, "hex": "#%02x%02x%02x" % (r, g, b)})
    calibration = {"calibrated": H is not None, "points": len((inp.get("pitch_calibration") or {}).get("points") or []),
                   "mean_error_m": round(calib_err, 2) if calib_err is not None else None}
    if pitch_info:
        calibration.update(pitch_info)
        calibration["calibrated"] = calibration["calibrated"] or pitch_info["frames_calibrated"] > 0
    if pitch_info:
        # Camera qui suit le jeu : equipes reconstituees avec la memoire des joueurs
        # hors champ, et mesures seulement quand l'equipe est presque entiere.
        mframes = [f for f in with_memory(frames, per_frame_H, worker.size) if f["calibrated"]]
        tactical = tactical_metrics(mframes, windows, L, W, min_players=TACTICAL_MIN_PLAYERS) if mframes else []
        counts = [sum(1 for p in f["points"] if not p["ball"] and p["team"] == tm) for f in mframes for tm in (0, 1)]
        if counts:
            calibration["players_per_team_median"] = float(np.median(counts))
    else:
        cal_frames = [f for f in frames if f["calibrated"]]
        tactical = tactical_metrics(cal_frames, windows, L, W) if cal_frames else []
    timings["total_s"] = round(time.time() - job_clock, 1)
    ball_frames = sum(1 for f in frames if any(p["ball"] for p in f["points"]))
    return {"simulated": False, "pitch": {"length": L, "width": W}, "tracks": tracks, "frames": frames,
            "frames_zip": frames_zip, "team_colors": team_colors, "play_windows": windows,
            "pitch_calibration": calibration, "tactical": tactical,
            "ball_frames": ball_frames, "total_frames": len(frames), "timings": timings}


# --- Test de detection automatique du terrain (calage image par image) ------
# Gabarit de terrain a 32 reperes (identifiants 0..31 des modeles publics de
# reperes de terrain), construit a la taille reelle du terrain du match :
# longueur et largeur variables, marquages aux tailles du reglement.
_PBW, _PBL, _GBW, _GBL, _CCR, _PSD = 40.32, 16.5, 18.32, 5.5, 9.15, 11.0


def pitch_keypoints(_PL=PITCH_L, _PW=PITCH_W):
    return [
        (0, 0), (0, (_PW - _PBW) / 2), (0, (_PW - _GBW) / 2), (0, (_PW + _GBW) / 2), (0, (_PW + _PBW) / 2), (0, _PW),
        (_GBL, (_PW - _GBW) / 2), (_GBL, (_PW + _GBW) / 2), (_PSD, _PW / 2),
        (_PBL, (_PW - _PBW) / 2), (_PBL, (_PW - _GBW) / 2), (_PBL, (_PW + _GBW) / 2), (_PBL, (_PW + _PBW) / 2),
        (_PL / 2, 0), (_PL / 2, _PW / 2 - _CCR), (_PL / 2, _PW / 2 + _CCR), (_PL / 2, _PW),
        (_PL - _PBL, (_PW - _PBW) / 2), (_PL - _PBL, (_PW - _GBW) / 2), (_PL - _PBL, (_PW + _GBW) / 2), (_PL - _PBL, (_PW + _PBW) / 2),
        (_PL - _PSD, _PW / 2), (_PL - _GBL, (_PW - _GBW) / 2), (_PL - _GBL, (_PW + _GBW) / 2),
        (_PL, 0), (_PL, (_PW - _PBW) / 2), (_PL, (_PW - _GBW) / 2), (_PL, (_PW + _GBW) / 2), (_PL, (_PW + _PBW) / 2), (_PL, _PW),
        (_PL / 2 - _CCR, _PW / 2), (_PL / 2 + _CCR, _PW / 2),
    ]


PITCH_KEYPOINTS = pitch_keypoints()


def pitch_size(inp):
    """Taille du terrain du match (metres), 105 x 68 par defaut."""
    pitch = inp.get("pitch") or {}
    return float(pitch.get("length", PITCH_L)), float(pitch.get("width", PITCH_W))


# Lignes du terrain (paires de reperes) pour dessiner le controle visuel.
PITCH_EDGES = [(0, 5), (0, 24), (5, 29), (24, 29), (13, 16), (1, 9), (9, 12), (12, 4), (2, 6), (6, 7), (7, 3),
               (25, 17), (17, 20), (20, 28), (26, 22), (22, 23), (23, 27)]


def _roboflow_keypoints(img, model, key):
    """Appelle le service en ligne Roboflow sur une image ; renvoie
    [(classe, x, y, confiance)] en pixels de `img`."""
    import base64
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    r = requests.post("https://detect.roboflow.com/%s" % model, params={"api_key": key},
                      data=base64.b64encode(jpg.tobytes()),
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=60)
    r.raise_for_status()
    out = []
    for det in r.json().get("predictions", []):
        for kp in det.get("keypoints", []):
            out.append((int(kp.get("class_id", -1)), float(kp["x"]), float(kp["y"]), float(kp.get("confidence", 0))))
    return out


def pitch_probe(inp):
    """Sur quelques images du match : detecte les reperes du terrain, calcule le
    calage de chaque image et renvoie la qualite obtenue + une image de controle
    (lignes du terrain redessinees a partir du calage)."""
    import base64
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set on this endpoint")
    model = inp.get("model", os.environ.get("PITCH_MODEL", "football-field-detection-f07vi/14"))
    min_conf = float(inp.get("min_confidence", 0.5))
    L, W = pitch_size(inp)
    src = download(inp["video_url"])
    cap = cv2.VideoCapture(src)
    results = []
    try:
        for t in inp.get("times") or []:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if not ok:
                results.append({"t": t, "error": "frame not readable"})
                continue
            h, w = frame.shape[:2]
            scale = 1280.0 / w if w > 1280 else 1.0
            img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            item = {"t": t}
            try:
                kps = _roboflow_keypoints(img, model, key)
            except Exception as e:
                item["error"] = "roboflow: %s" % str(e)[:200]
                results.append(item)
                continue
            good = [k for k in kps if k[3] >= min_conf and 0 <= k[0] < len(PITCH_KEYPOINTS)]
            item.update(keypoints_detected=len(kps), keypoints_confident=len(good))
            vis = img.copy()
            for c, x, y, conf in good:
                cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            auto = auto_fit(img, good, L, W) if len(good) >= 4 else None
            if auto:
                H, score = auto
                item.update(calibrated=bool(score["line_hit"] >= AUTO_FIT_GOOD), line_hit=score["line_hit"],
                            mean_error_m=None, mean_dist_px=score["mean_dist_px"])
                vis = pitch_fit.draw_overlay(vis, H, length=L, width=W)
            else:
                item["calibrated"] = False
            small = cv2.resize(vis, (960, int(vis.shape[0] * 960 / vis.shape[1])))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            item["preview_jpg_base64"] = base64.b64encode(jpg.tobytes()).decode()
            results.append(item)
    finally:
        cap.release()
        os.remove(src)
    done = [r for r in results if r.get("calibrated")]
    return {"task": "pitch_probe", "model": model, "frames": len(results), "calibrated_frames": len(done),
            "median_error_m": None,
            "median_line_hit": round(float(np.median([r["line_hit"] for r in results if r.get("line_hit") is not None])), 3)
            if any(r.get("line_hit") is not None for r in results) else None,
            "results": results}


AUTO_FIT_GOOD = float(os.environ.get("AUTO_FIT_GOOD", "0.8"))


def auto_fit(frame, kps, L=PITCH_L, W=PITCH_W):
    """Calage automatique d'une image : homographie de depart a partir des
    reperes (au moins 4 fiables), puis ajustement sur les lignes blanches.
    Renvoie (H terrain -> image, score) ou None."""
    H0, inliers = pitch_fit.homography_from_keypoints(kps, pitch_keypoints(L, W))
    if H0 is None:
        return None
    try:
        return pitch_fit.fit_best(frame, H0, length=L, width=W)
    except Exception:
        return None


def label_batch(inp):
    """Prepare un lot d'images a corriger a la main (entrainement de notre
    propre modele de reperes du terrain) : images reparties sur les periodes
    de jeu, chacune avec les reperes suggeres (si une cle Roboflow est
    configuree). Tout est depose en un zip (images + manifest.json) a
    l'adresse signee frames_upload_url."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    model = inp.get("model", os.environ.get("PITCH_MODEL", "football-field-detection-f07vi/14"))
    count = int(inp.get("count", 50))
    L, W = pitch_size(inp)
    template_kp = pitch_keypoints(L, W)
    src = download(inp["video_url"])
    cap = cv2.VideoCapture(src)
    dur = video_end(cap) or 0
    windows = play_windows(inp, 0.0, dur) if dur else [[0.0, 0.0]]
    total = sum(we - ws for ws, we in windows)
    # Instants regulierement espaces sur le temps de jeu (jamais deux images voisines).
    # Decalage propre a chaque lot : deux lots sur le meme match ne prennent
    # jamais les memes instants (sinon on etiquetterait deux fois les memes images).
    import random
    phase = float(inp["phase"]) if inp.get("phase") is not None else random.random()
    times = []
    for k in range(count):
        pos = (k + phase) * total / count
        for ws, we in windows:
            if pos <= we - ws:
                times.append(round(ws + pos, 2))
                break
            pos -= we - ws
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED)
    items = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            fh, fw = frame.shape[:2]
            suggestions, err = [], None
            if key:
                try:
                    for c, x, y, conf in _roboflow_keypoints(frame, model, key):
                        if 0 <= c < len(PITCH_KEYPOINTS):
                            suggestions.append({"id": c, "u": round(x / fw, 4), "v": round(y / fh, 4),
                                                "confidence": round(conf, 3)})
                except Exception as e:
                    err = str(e)[:200]
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            name = "label_%09d.jpg" % int(round(t * 1000))
            zf.writestr(name, jpg.tobytes())
            item = {"t": t, "file": name, "width": fw, "height": fh,
                    "suggestions": suggestions, "suggestion_error": err}
            # Calage automatique : depart grossier (reperes suggeres) puis
            # ajustement precis sur les lignes blanches de la pelouse.
            auto = auto_fit(frame, [(s["id"], s["u"] * fw, s["v"] * fh, s["confidence"]) for s in suggestions], L, W)
            if auto:
                H, score = auto
                kp = cv2.perspectiveTransform(np.float32([template_kp]), H)[0]
                item["auto"] = {
                    "line_hit": score["line_hit"], "mean_dist_px": score["mean_dist_px"],
                    "good": score["line_hit"] >= AUTO_FIT_GOOD,
                    "keypoints": [{"id": i, "u": round(float(x) / fw, 4), "v": round(float(y) / fh, 4),
                                   "visible": bool(0 <= x < fw and 0 <= y < fh)} for i, (x, y) in enumerate(kp)],
                }
                prev = pitch_fit.draw_overlay(frame, H, length=L, width=W)
                ok, pj = cv2.imencode(".jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 80])
                item["auto"]["preview_file"] = name.replace(".jpg", "_fit.jpg")
                zf.writestr(item["auto"]["preview_file"], pj.tobytes())
            items.append(item)
    finally:
        cap.release()
        os.remove(src)
    template = [{"id": i, "pitch": [x, y]} for i, (x, y) in enumerate(template_kp)]
    zf.writestr("manifest.json", json.dumps({"pitch_template_m": {"length": L, "width": W},
                                             "keypoints": template, "items": items}))
    zf.close()
    data = buf.getvalue()
    r = requests.put(inp["frames_upload_url"], data=data, timeout=600,
                     headers={"Content-Type": "application/zip", "x-upsert": "true"})
    r.raise_for_status()
    return {"task": "label_batch", "count": len(items), "bytes": len(data), "phase": round(phase, 3),
            "auto_fitted": sum(1 for i in items if i.get("auto")),
            "auto_good": sum(1 for i in items if (i.get("auto") or {}).get("good")),
            "with_suggestions": sum(1 for i in items if i["suggestions"])}


PROXY_WIDTH = int(os.environ.get("PROXY_WIDTH", "1920"))
PROXY_FPS = int(os.environ.get("PROXY_FPS", "15"))
# L'image de base contient un ffmpeg minimal (sans encodeur H.264) en tete du
# PATH : on utilise celui du systeme, installe par apt, s'il existe.
FFMPEG = os.environ.get("FFMPEG_BIN") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg")


def _ffmpeg_encode(src, dst, width, fps, encoder):
    """Re-encode la video en plus petit (largeur `width`, `fps` images/s, sans son)."""
    import subprocess
    vf = "scale='min(%d,iw)':-2,fps=%d" % (width, fps)
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28", "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-an",
           *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s: %s" % (encoder, r.stderr.strip()[-300:]))


def prepare(inp):
    """Tache unique apres l'envoi : cree une copie allegee de la video
    (1920 de large, 15 images/s, sans son) et l'envoie a l'adresse signee
    `proxy_upload_url`. Toutes les analyses suivantes peuvent utiliser cette
    copie : telechargement, lecture et extraction bien plus rapides."""
    timings = {}
    clock = job_clock = time.time()
    src = download(inp["video_url"])
    timings["download_s"] = round(time.time() - clock, 1)
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = video_end(cap)
    cap.release()
    width = int(inp.get("proxy_width", PROXY_WIDTH))
    fps = int(inp.get("proxy_fps", PROXY_FPS))
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    clock = time.time()
    encoder, errors = None, []
    for enc in ("h264_nvenc", "libx264"):   # puce video du GPU d'abord, sinon processeur
        try:
            _ffmpeg_encode(src, dst, width, fps, enc)
            encoder = enc
            break
        except Exception as e:
            errors.append(str(e)[:200])
    timings["encode_s"] = round(time.time() - clock, 1)
    if encoder is None:
        os.remove(src)
        raise RuntimeError("proxy encoding failed: " + " | ".join(errors))
    cap = cv2.VideoCapture(dst)
    pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    size_in, size_out = os.path.getsize(src), os.path.getsize(dst)
    os.remove(src)
    clock = time.time()
    with open(dst, "rb") as f:
        r = requests.put(inp["proxy_upload_url"], data=f, timeout=1800,
                         headers={"Content-Type": "video/mp4", "x-upsert": "true"})
    r.raise_for_status()
    timings["upload_s"] = round(time.time() - clock, 1)
    os.remove(dst)
    timings["total_s"] = round(time.time() - job_clock, 1)
    return {"task": "prepare", "encoder": encoder, "original": {"width": w, "height": h, "bytes": size_in},
            "proxy": {"width": pw, "height": ph, "fps": fps, "bytes": size_out},
            "duration_s": round(duration, 2) if duration else None, "timings": timings}


def notify(inp, **fields):
    """Message intermediaire vers l'application (n'interrompt jamais le travail)."""
    try:
        body = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
        body.update(fields)
        requests.post(inp["callback_url"], json=body, timeout=60)
    except Exception:
        pass


def handler(job):
    inp = job["input"]
    cb = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
    if inp.get("task") == "label_batch":
        try:
            cb.update(status="label_batch_ready", result=label_batch(inp))
        except Exception as e:
            cb.update(status="label_batch_failed", error=str(e)[:900])
        requests.post(inp["callback_url"], json=cb, timeout=120)
        return {"status": cb["status"]}
    if inp.get("task") == "pitch_probe":
        # Test de detection automatique du terrain : message de retour distinct.
        try:
            cb.update(status="probe_ready", result=pitch_probe(inp))
        except Exception as e:
            cb.update(status="probe_failed", error=str(e)[:900])
        requests.post(inp["callback_url"], json=cb, timeout=120)
        return {"status": cb["status"]}
    if inp.get("task") == "prepare":
        # Copie allegee : message de retour distinct ("proxy_ready" / "proxy_failed")
        # pour ne jamais etre confondu avec la fin d'une analyse.
        try:
            result = prepare(inp)
            cb.update(status="proxy_ready", result=result)
        except Exception as e:
            cb.update(status="proxy_failed", error=str(e)[:900])
        requests.post(inp["callback_url"], json=cb, timeout=120)
        return {"status": cb["status"]}
    try:
        result = process(inp)
        cb.update(status="completed", result=result, stats={"frames": len(result["frames"]), "tracks": len(result.get("tracks", []))})
    except Exception as e:
        cb.update(status="failed", error=str(e)[:900])
    requests.post(inp["callback_url"], json=cb, timeout=120)
    return {"status": cb["status"], "frames": len(cb.get("result", {}).get("frames", []))}


runpod.serverless.start({"handler": handler})
