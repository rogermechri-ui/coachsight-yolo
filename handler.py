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

import cv2
import numpy as np
import requests
import runpod
from ultralytics import YOLO

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


def track(cap, windows, fps_out, timings):
    """Detection + suivi des joueurs et du ballon dans chaque periode de jeu.
    Les positions sont gardees en coordonnees d'image normalisees (0..1) :
    la conversion en metres se fait ensuite (calage du terrain si fourni)."""
    raw = []      # (t, [(trackId, u, v, ball, color)])
    votes = {}    # trackId -> numeros lus (vote majoritaire en fin de traitement)
    frame_i = 0
    t_read = t_detect = t_color = t_ocr = 0.0
    MODEL.predictor = None   # repart d'un suivi vierge (cas d'une 2e tentative)
    for ws, we in windows:
        clock = time.time()
        for t, frame in iter_frames(cap, ws, we, fps_out):
            t_read += time.time() - clock
            h, w = frame.shape[:2]
            c = time.time()
            res = MODEL.track(frame, persist=True, classes=[0, 32], conf=0.25, verbose=False)[0]
            t_detect += time.time() - c
            pts = []
            if res.boxes is not None:
                ids = res.boxes.id.tolist() if res.boxes.id is not None else [None] * len(res.boxes)
                for box, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), ids):
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
    clock = time.time()
    raw, votes = track(cap, windows, fps_out, timings)
    expected = int(sum(we - ws for ws, we in windows) * fps_out) + len(windows)
    # Lecture directe interrompue (coupure reseau) : on recommence a partir
    # d'une copie complete de la video pour ne rien perdre.
    if mode == "stream" and len(raw) < 0.95 * expected:
        timings["stream_fallback"] = "%d/%d images lues" % (len(raw), expected)
        cap.release()
        tmp = download(url)
        cap = cv2.VideoCapture(tmp)
        timings["video_mode"] = "download"
        raw, votes = track(cap, windows, fps_out, timings)
    timings["tracking_s"] = round(time.time() - clock, 1)
    cap.release()
    if tmp:
        os.remove(tmp)

    # Positions en metres ; avec calage, tout ce qui est hors du terrain
    # (public, bancs, arbitres de touche) est ecarte.
    margin = 2.0
    placed = []
    for tt, pts in raw:
        keep = []
        for tid, u, v, ball, col in pts:
            x, y = to_pitch(H, u, v, L, W)
            if H is not None and not (-margin <= x <= L + margin and -margin <= y <= W + margin):
                continue
            keep.append((tid, x, y, ball, col))
        placed.append((tt, keep))

    cols = np.array([p[4] for _, pts in placed for p in pts if p[4] is not None], dtype=np.float32)
    centers = None
    if len(cols) >= 4:
        _, _, centers = cv2.kmeans(cols, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    # Numero retenu par piste : celui le plus souvent lu (vote majoritaire).
    numbers = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    tracks = []
    seen = {}
    frames = []
    for tt, pts in placed:
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
        frames.append({"t": tt, "points": out})
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
    tactical = tactical_metrics(frames, windows, L, W) if H is not None else []
    timings["total_s"] = round(time.time() - job_clock, 1)
    ball_frames = sum(1 for f in frames if any(p["ball"] for p in f["points"]))
    return {"simulated": False, "pitch": {"length": L, "width": W}, "tracks": tracks, "frames": frames,
            "frames_zip": frames_zip, "team_colors": team_colors, "play_windows": windows,
            "pitch_calibration": calibration, "tactical": tactical,
            "ball_frames": ball_frames, "total_frames": len(frames), "timings": timings}


PROXY_WIDTH = int(os.environ.get("PROXY_WIDTH", "1920"))
PROXY_FPS = int(os.environ.get("PROXY_FPS", "15"))


def _ffmpeg_encode(src, dst, width, fps, encoder):
    """Re-encode la video en plus petit (largeur `width`, `fps` images/s, sans son)."""
    import subprocess
    vf = "scale='min(%d,iw)':-2,fps=%d" % (width, fps)
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "28", "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-an",
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
