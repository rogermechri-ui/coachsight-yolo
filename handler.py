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
# Processus de calage en parallele (tranches du match).
PITCH_PROCS = int(os.environ.get("PITCH_PROCS", str(max(1, min(6, (os.cpu_count() or 2) - 1)))))
# Recalage fin sur les lignes une image analysee sur N (les autres : mouvement seul).
PITCH_FULL_EVERY = int(os.environ.get("PITCH_FULL_EVERY", "2"))
# Detection : on retire les bandes noires des videos Veo (image 16:9 dans un
# cadre 21:9) et on analyse a une taille suffisante pour voir les joueurs
# lointains (640 = taille par defaut de YOLO : la moitie des joueurs echappent).
CROP_BARS = os.environ.get("CROP_BARS", "1") != "0"
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))
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
    # Clarte : 90e centile plutot que la mediane. Un joueur lointain en maillot
    # clair se melange aux panneaux sombres derriere lui ; sa mediane le classait
    # dans l'equipe en maillot fonce.
    return np.array([np.percentile(keep[:, 0], 90), np.median(keep[:, 1]), np.median(keep[:, 2])],
                    dtype=np.float32)


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


# Decodage de la video par ffmpeg, sur la carte graphique quand c'est possible
# (NVDEC) : le decodage CPU d'une video 3440 px etait le poste le plus variable
# d'une machine RunPod a l'autre (6 a 30 min par match). "auto" essaie le GPU
# puis ffmpeg CPU puis OpenCV ; "cv2" revient a l'ancienne lecture.
# Mesure du 8/10 (job 3028278c, 7 decodages NVDEC en parallele) : pas plus rapide
# que le CPU (lecture 1 780 s) et les images reduites a 1 920 px degradent la
# reconnaissance des equipes. Par defaut on garde donc OpenCV ; "auto" reste
# disponible pour de nouveaux essais.
VIDEO_DECODER = os.environ.get("VIDEO_DECODER", "cv2")
DETECT_WIDTH = int(os.environ.get("DETECT_WIDTH", "0"))      # 0 = pleine resolution pour la detection
_DECODER_OK = {}        # mode -> True/False, appris au premier essai (par processus)


def _video_size(source):
    """(largeur, hauteur, codec) de la video ; codec = 'h264', 'hevc', ... ou ''."""
    cap = cv2.VideoCapture(source)
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    tag = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).lower()
    codec = {"avc1": "h264", "h264": "h264", "x264": "h264", "hev1": "hevc", "hvc1": "hevc",
             "hevc": "hevc", "av01": "av1"}.get(tag.strip("\x00 "), "")
    return w, h, codec


_CUVID = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "av1": "av1_cuvid"}


def _ffmpeg_frames(source, start, end, fps, width, gpu):
    """Images (t, BGR) de `source` entre start et end a `fps`, reduites a `width`
    px de large, via ffmpeg (NVDEC + redimensionnement sur la carte si gpu)."""
    import subprocess
    w0, h0, codec = _video_size(source)
    if not w0 or not h0:
        raise RuntimeError("video size unknown")
    if width and width < w0:
        w, h = width, int(round(h0 * width / w0 / 2)) * 2
    else:
        w, h = w0, h0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    vf = ["fps=%g:round=near" % fps]
    if gpu:
        # Decodeur NVDEC (cuvid) : il reduit lui-meme l'image (-resize), les images
        # arrivent en memoire centrale ; aucun filtre CUDA necessaire.
        if codec not in _CUVID:
            raise RuntimeError("no NVDEC decoder for codec %r" % codec)
        cmd += ["-c:v", _CUVID[codec]]
        if (w, h) != (w0, h0):
            cmd += ["-resize", "%dx%d" % (w, h)]
    elif (w, h) != (w0, h0):
        vf.append("scale=%d:%d" % (w, h))
    cmd += ["-ss", "%.3f" % start, "-t", "%.3f" % max(0.0, end - start + 0.5 / fps), "-i", source,
            "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    n_bytes = w * h * 3
    k = 0
    try:
        while True:
            parts, got = [], 0
            while got < n_bytes:              # un tube rend parfois moins que demande
                chunk = p.stdout.read(n_bytes - got)
                if not chunk:
                    break
                parts.append(chunk)
                got += len(chunk)
            buf = b"".join(parts)
            if len(buf) < n_bytes:            # fin de la video (ou erreur ffmpeg)
                if k == 0:
                    err = p.stderr.read().decode(errors="replace").strip()
                    raise RuntimeError("ffmpeg produced no frame: " + err[-300:])
                break
            t = start + k / fps
            if t > end + 1e-6:
                break
            yield t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        # Arret avant la fin : ffmpeg peut etre bloque en ecriture ; on le tue
        # avant de fermer les tubes (sinon interblocage).
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.stdout.close()
            p.stderr.close()
        except Exception:
            pass


def read_frames(source, start, end, fps, width=None):
    """Images (t, BGR) toutes les 1/fps s entre start et end : ffmpeg sur la carte
    graphique, sinon ffmpeg CPU, sinon OpenCV (ancienne lecture). Le premier mode
    qui marche est garde pour la suite du processus."""
    modes = {"auto": ["gpu", "cpu", "cv2"], "gpu": ["gpu", "cv2"], "cpu": ["cpu", "cv2"], "cv2": ["cv2"]}
    for mode in modes.get(VIDEO_DECODER, modes["auto"]):
        if _DECODER_OK.get(mode) is False:
            continue
        if mode == "cv2":
            cap = cv2.VideoCapture(source)
            try:
                for t, frame in iter_frames(cap, start, end, fps):
                    if width and frame.shape[1] > width:
                        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])),
                                           interpolation=cv2.INTER_AREA)
                    yield t, frame
            finally:
                cap.release()
            return
        gen = _ffmpeg_frames(source, start, end, fps, width, gpu=(mode == "gpu"))
        try:
            first = next(gen)
        except Exception as e:
            if _DECODER_OK.get(mode) is None:
                print("video decoder %s unavailable: %s" % (mode, str(e)[:200]), flush=True)
            _DECODER_OK[mode] = False
            continue
        if _DECODER_OK.get(mode) is None:
            print("video decoder: %s" % mode, flush=True)
        _DECODER_OK[mode] = True
        _DECODER_OK["used"] = mode
        yield first
        yield from gen
        return


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


def download(url, attempts=5):
    """Copie locale de la video. Si la connexion coupe en cours de route (gros
    fichiers de plusieurs Go), on reprend la ou on s'etait arrete (en-tete Range)
    au lieu d'echouer."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    done, total = 0, None
    for k in range(attempts):
        headers = {"Range": "bytes=%d-" % done} if done else {}
        try:
            with requests.get(url, stream=True, timeout=600, headers=headers) as r:
                r.raise_for_status()
                if done and r.status_code != 206:      # le serveur ignore Range : on repart de zero
                    done = 0
                if total is None:
                    size = r.headers.get("Content-Length")
                    total = (int(size) + done) if size else None
                with open(tmp, "ab" if done else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
            if total is None or done >= total:
                return tmp
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print("download interrupted at %d bytes (%s), retry %d" % (done, type(e).__name__, k + 1), flush=True)
        time.sleep(2 + 3 * k)
    raise RuntimeError("video download failed after %d attempts (%d bytes of %s)" % (attempts, done, total))


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


def _pitch_slice(camera, L, W, source, slices, fps_out, k, out):
    """Processus de calage d'une tranche du match : il lit lui-meme sa partie de
    la video (decodage en parallele) et renvoie l'homographie de chaque image
    analysee, reperee par son instant t (memes instants que la detection)."""
    cv2.setNumThreads(1)
    tracker = pitch_track.PitchTracker(camera, L, W)
    H, size, err, n_full = {}, None, None, 0
    t0 = time.time()
    try:
        last_w = None
        for ws, we, wid, a, b in slices:
            if last_w is not None and wid != last_w:      # nouvelle periode : la camera a pu bouger
                tracker.p = None
                tracker.feat = None
            last_w = wid
            for j, (t, frame) in enumerate(read_frames(source, ws, we, fps_out * k, width=pitch_track.WORK_W)):
                img = pitch_track.to_work(frame)
                size = (img.shape[1], img.shape[0])
                if j % k:
                    tracker.motion(img)
                    continue
                if n_full % PITCH_FULL_EVERY == 0 or tracker.p is None:
                    h, s = tracker.full(img)
                else:
                    tracker.motion(img)
                    h, s = tracker.current(), None
                n_full += 1
                if a - 1e-6 <= t <= b + 1e-6:      # les secondes d'avant ne servent qu'a demarrer
                    H[round(t, 2)] = (None if h is None else h.tolist(), s)
    except Exception as e:  # le calage ne doit jamais faire echouer l'analyse
        err = "%s: %s" % (type(e).__name__, str(e)[:200])
    st = dict(tracker.stats)
    st["wall_s"] = time.time() - t0
    out.put({"H": H, "size": size, "error": err, "stats": st})


class PitchWorker:
    """Calage du terrain en parallele de la detection des joueurs (GPU).
    Le match est coupe en PITCH_PROCS tranches de temps de jeu ; chaque processus
    relit sa tranche de la video et la cale seul (il se cale au debut de sa
    tranche). Resultat : l'homographie terrain -> image de chaque image analysee."""

    def __init__(self, camera, L, W, windows, source, fps_out):
        import multiprocessing as mp
        ctx = mp.get_context("fork")     # pas "spawn" : il rechargerait tout le moteur
        total = sum(we - ws for ws, we in windows)
        n = max(1, min(PITCH_PROCS, int(total // 120) or 1))    # au moins 2 min par tranche
        k = max(1, int(round(MOTION_FPS / fps_out)))
        step = total / n
        # Tranches : chaque processus recoit des morceaux (debut lecture, fin lecture,
        # periode, debut garde, fin garde). La lecture commence 3 s avant le debut
        # garde pour que le mouvement de la camera soit deja suivi.
        slices = [[] for _ in range(n)]
        played = 0.0
        for wid, (ws, we) in enumerate(windows):
            for c in range(n):
                a_play, b_play = c * step, (c + 1) * step
                lo, hi = max(a_play, played), min(b_play, played + (we - ws))
                if hi - lo > 1e-6:
                    a, b = ws + (lo - played), ws + (hi - played)
                    slices[c].append((max(ws, a - 3.0) if a > ws else ws, b, wid, a, b))
            played += we - ws
        # Les instants analyses doivent etre ceux de la detection : on lit chaque
        # morceau depuis le debut de sa periode n'est pas necessaire, iter_frames
        # part de ws ; on aligne donc le debut de lecture sur la grille 1/fps_out.
        grid = 1.0 / (fps_out * k)
        aligned = []
        for sl in slices:
            al = []
            for ws_read, b, wid, a, bb in sl:
                w0 = windows[wid][0]
                ws_read = w0 + np.floor((ws_read - w0) / (1.0 / fps_out) + 1e-9) * (1.0 / fps_out)
                al.append((ws_read, b, wid, a, bb))
            aligned.append(al)
        self.out = ctx.Queue()
        self.procs = [ctx.Process(target=_pitch_slice, args=(camera, L, W, source, sl, fps_out, k, self.out),
                                  daemon=True) for sl in aligned if sl]
        for pr in self.procs:
            pr.start()
        self.H, self.size, self.error, self.stats = {}, None, None, {}

    def finish(self):
        errors = []
        walls = []
        for _ in self.procs:
            try:
                r = self.out.get(timeout=3600)
            except Exception:
                errors.append("pitch process lost")
                continue
            self.H.update({t: (None if h is None else np.array(h), s) for t, (h, s) in r["H"].items()})
            self.size = self.size or r["size"]
            if r["error"]:
                errors.append(r["error"])
            st = r["stats"]
            walls.append(st.pop("wall_s", 0))
            for key, v in st.items():
                self.stats[key] = self.stats.get(key, 0) + v
        for pr in self.procs:
            pr.join(timeout=10)
        self.error = "; ".join(errors) or None
        self.stats["processes"] = len(self.procs)
        self.stats["slowest_s"] = round(max(walls), 1) if walls else 0
        return self.H


def track(source, windows, fps_out, timings):
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
    crop = None     # zone utile de l'image (sans les bandes noires), fixee a la 1re image
    for ws, we in windows:
        clock = time.time()
        for t, frame in read_frames(source, ws, we, fps_out, width=DETECT_WIDTH):
            t_read += time.time() - clock
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


CAMERA_SEGMENTS = int(os.environ.get("CAMERA_SEGMENTS", "4"))      # passages de la video etudies
CAMERA_SEGMENT_S = float(os.environ.get("CAMERA_SEGMENT_S", "30"))  # duree de chaque passage
CAMERA_FPS = 6.0


def camera_frames(source, windows, n=CAMERA_SEGMENTS, dur=CAMERA_SEGMENT_S, fps=CAMERA_FPS):
    """Images (6/s) de n passages de `dur` secondes repartis dans les periodes de
    jeu, pour estimer la position de la camera (vues variees : panoramiques, zooms)."""
    total = sum(we - ws for ws, we in windows)
    if total <= 0:
        return
    marks = [total * (k + 0.5) / n for k in range(n)]
    for m in marks:
        acc = 0.0
        for ws, we in windows:
            if acc + (we - ws) > m:
                a = min(ws + (m - acc), max(ws, we - dur))
                for _, frame in read_frames(source, a, min(we, a + dur), fps, width=pitch_track.WORK_W):
                    yield frame
                break
            acc += we - ws


def estimate_camera(source, windows, L, W):
    """Position de la camera estimee sur la video elle-meme (camera fixe type Veo)."""
    return pitch_track.estimate_camera(camera_frames(source, windows), L, W,
                                       log=lambda s: print(s, flush=True))


def pitch_camera(inp):
    """Position de la camera (metres) : pitch.camera envoye par l'application
    ({x, y, z} ou [x, y, z]), sinon PITCH_CAMERA ; None = pas de calage auto.
    "auto" (ou rien du tout) : a estimer sur la video."""
    cam = (inp.get("pitch") or {}).get("camera")
    if cam == "auto":
        return "auto"
    try:
        if isinstance(cam, dict):
            return np.array([float(cam["x"]), float(cam["y"]), float(cam["z"])])
        if isinstance(cam, (list, tuple)) and len(cam) == 3:
            return np.array([float(c) for c in cam])
        if PITCH_CAMERA == "auto":
            return "auto"
        if PITCH_CAMERA:
            return np.array([float(c) for c in PITCH_CAMERA.split(",")])
    except (KeyError, TypeError, ValueError):
        return None
    return "auto" if cam is None and AUTO_CAMERA else None


AUTO_CAMERA = os.environ.get("AUTO_CAMERA", "1") == "1"   # sans position connue : estimer sur la video


# Memoire des joueurs sortis de l'image (camera Veo qui suit le ballon : on ne
# voit qu'une partie du terrain a chaque instant).
MEMORY_S = float(os.environ.get("PLAYER_MEMORY_S", "6"))
TACTICAL_MIN_PLAYERS = int(os.environ.get("TACTICAL_MIN_PLAYERS", "8"))
TEAM_HUE_MIN = float(os.environ.get("TEAM_HUE_MIN", "20"))   # ecart de teinte (Lab a,b) accepte


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


KEEPER_GAP_M = 8.0
SIDE_SWITCH_MIN_S = 50 * 60     # une periode plus longue contient sans doute les deux mi-temps
MIN_SPREAD_M = 6.0              # six joueurs ou plus tiennent toujours dans plus de 6 m x 6 m


def positions_plausible(pts):
    """Faux quand le calage de l'image est degenere : au moins six joueurs
    projetes dans un carre de MIN_SPREAD_M de cote (tous au meme endroit).
    Observe sur l'original 3440 px : des tranches entieres avec 10 a 18 joueurs
    a x = 53 m, qui tiraient la ligne defensive vers le milieu du terrain."""
    if len(pts) < 6:
        return True
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs) >= MIN_SPREAD_M or max(ys) - min(ys) >= MIN_SPREAD_M


def split_at_side_switch(frames, windows):
    """Les equipes changent de cote a la mi-temps. Une periode de plus de 50 min
    (match entier envoye d'un bloc) est coupee au moment ou l'ordre gauche/droite
    des deux equipes s'inverse durablement ; sinon elle reste entiere."""
    out = []
    for ws, we in windows:
        if we - ws < SIDE_SWITCH_MIN_S:
            out.append([ws, we])
            continue
        ts, sides = [], []
        for f in frames:
            if not ws <= f["t"] <= we:
                continue
            xs = {tm: [p["x"] for p in f["points"] if not p["ball"] and p["team"] == tm] for tm in (0, 1)}
            if len(xs[0]) >= 4 and len(xs[1]) >= 4:
                ts.append(f["t"])
                sides.append(1 if np.mean(xs[0]) < np.mean(xs[1]) else -1)
        if len(sides) < 200:
            out.append([ws, we])
            continue
        s = np.array(sides)
        before = np.cumsum(s)                      # somme des signes jusqu'a i inclus
        total = before[-1]
        # Accord si l'on coupe apres i : |somme avant| + |somme apres|.
        gain = np.abs(before[:-1]) + np.abs(total - before[:-1])
        lo, hi = int(0.15 * len(s)), int(0.85 * len(s))
        i = lo + int(np.argmax(gain[lo:hi]))
        n1, n2 = i + 1, len(s) - i - 1
        s1, s2 = before[i], total - before[i]
        # Coupure retenue si les deux parties ont chacune un cote net et oppose :
        # au moins 60 % d'accord et un ecart bien au-dela du hasard (3 sigmas).
        # (Mesure sur ce match : 83 % en 1re mi-temps, 68 % en 2e.)
        if (s1 * s2 < 0 and abs(s1) / n1 >= 0.2 and abs(s2) / n2 >= 0.2
                and abs(s1) >= 3 * np.sqrt(n1) and abs(s2) >= 3 * np.sqrt(n2)):
            cut = round((ts[i] + ts[i + 1]) / 2, 1)
            out += [[ws, cut], [cut, we]]
        else:
            out.append([ws, we])
    return out


def tactical_metrics(frames, windows, L, W, min_players=6):
    """Indicateurs par periode et par equipe (groupes 0 et 1), a partir des
    positions en metres (necessite un terrain cale) :
    hauteur de la ligne defensive, largeur du bloc, longueur du bloc (compacite)
    et part du jeu dans chaque tiers. Le gardien (joueur le plus proche de son
    but) est exclu des mesures de bloc."""
    out = []
    periods = []
    for ws, we in split_at_side_switch(frames, windows):
        fr = [f for f in frames if ws <= f["t"] <= we]
        per_team = {0: [], 1: []}
        votes = []          # par image ou les deux equipes sont visibles : equipe 0 a gauche ?
        for f in fr:
            if not positions_plausible([(p["x"], p["y"]) for p in f["points"] if not p["ball"]]):
                continue        # calage degenere : tous les joueurs au meme endroit
            xs = {team: [(p["x"], p["y"]) for p in f["points"] if not p["ball"] and p["team"] == team] for team in (0, 1)}
            for team in (0, 1):
                if len(xs[team]) >= min_players:
                    per_team[team].append(xs[team])
            if len(xs[0]) >= 4 and len(xs[1]) >= 4:
                votes.append(np.mean([p[0] for p in xs[0]]) < np.mean([p[0] for p in xs[1]]))
        periods.append((ws, we, per_team, votes))
    sides = side_per_period(periods)
    for (ws, we, per_team, votes), (left, conf) in zip(periods, sides):
        if not per_team[0] or not per_team[1]:
            out.append({"window": [ws, we], "frames_used": 0, "teams": {}})
            continue
        teams = {}
        for team in (0, 1):
            own_left = team == left
            lines, widths, lengths, thirds = [], [], [], [0, 0, 0]
            for xs in per_team[team]:
                d = sorted(((x if own_left else L - x), y) for x, y in xs)   # distance a son propre but
                # Gardien : joueur le plus en retrait, nettement detache du suivant
                # (le gardien en maillot different est deja sans equipe).
                outfield = d[1:] if d[1][0] - d[0][0] >= KEEPER_GAP_M else d
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
        out.append({"window": [ws, we], "frames_used": min(len(per_team[0]), len(per_team[1])), "teams": teams,
                    "side_confidence_pct": round(100 * abs(conf)),
                    "side_source": "other_half" if conf < 0 else "frames"})
    return out


def side_per_period(periods):
    """Quel but defend l'equipe 0 dans chaque periode. Vote image par image (les
    deux equipes visibles) ; confiance = part des images d'accord. Les equipes
    changent de cote a la mi-temps : la periode la plus sure fixe les autres,
    en alternant a chaque coupure de plus de HALF_GAP_S (pause). Renvoie
    [(equipe a gauche, confiance)] ; sans vote fiable, cote tire des positions
    moyennes."""
    res = []
    for ws, we, per_team, votes in periods:
        if len(votes) >= 50:
            share = float(np.mean(votes))
            res.append([0 if share >= 0.5 else 1, abs(share - 0.5) * 2])
        elif per_team[0] and per_team[1]:
            mean_x = {t: float(np.mean([np.mean([p[0] for p in xs]) for xs in per_team[t]])) for t in (0, 1)}
            res.append([0 if mean_x[0] <= mean_x[1] else 1, 0.0])
        else:
            res.append([0, 0.0])
    if len(periods) < 2:
        return [tuple(r) for r in res]
    # Numero de mi-temps de chaque periode (change apres une pause).
    half = [0]
    for (_, prev_end, _, _), (ws, _, _, _) in zip(periods, periods[1:]):
        # Pause longue, ou coupure posee par split_at_side_switch (periodes jointives).
        half.append(half[-1] + (1 if ws - prev_end >= HALF_GAP_S or ws == prev_end else 0))
    best = int(np.argmax([r[1] for r in res]))
    if res[best][1] >= SIDE_MIN_CONF:
        for i, r in enumerate(res):
            if r[1] < SIDE_MIN_CONF:
                flip = (half[i] - half[best]) % 2
                r[0] = res[best][0] ^ flip
                r[1] = -res[best][1]        # negatif : cote deduit de l'autre mi-temps
    return [tuple(r) for r in res]


HALF_GAP_S = 5 * 60       # coupure entre deux periodes consideree comme une mi-temps
SIDE_MIN_CONF = 0.4       # 70 % des images d'accord


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
    timings["cpus"] = os.cpu_count()
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
    camera_info = {"source": "given"} if camera is not None else {"source": "none"}
    if isinstance(camera, str):          # "auto" : position estimee sur la video
        clock = time.time()
        camera, cinfo = estimate_camera(tmp or url, windows, L, W)
        camera_info = {"source": "auto", **cinfo}
        timings["camera_s"] = round(time.time() - clock, 1)
    clock = time.time()
    worker = PitchWorker(camera, L, W, windows, tmp or url, fps_out) if camera is not None else None
    raw, votes = track(tmp or url, windows, fps_out, timings)
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
            worker = PitchWorker(camera, L, W, windows, tmp, fps_out)
        raw, votes = track(tmp or url, windows, fps_out, timings)
    timings["tracking_s"] = round(time.time() - clock, 1)
    timings["video_decoder"] = _DECODER_OK.get("used", "cv2")
    per_frame_H, pitch_info = {}, None
    if worker:
        c = time.time()
        by_t = worker.finish()
        per_frame_H = {i: by_t.get(round(t, 2), (None, None)) for i, (t, _) in enumerate(raw)}
        timings["pitch_wait_s"] = round(time.time() - c, 1)     # attente du calage apres le suivi
        timings["pitch_slowest_s"] = worker.stats.get("slowest_s")
        st = worker.stats
        timings["pitch_s"] = round(st.get("time_s", 0), 1)          # temps de calcul cumule (toutes tranches)
        timings["pitch_global_s"] = round(st.get("global_s", 0), 1)
        timings["pitch_processes"] = st.get("processes", 0)
        n_cal = sum(1 for H, _ in per_frame_H.values() if H is not None)
        pitch_info = {"method": "camera_tracking", "camera": [round(float(x), 2) for x in camera],
                      "frames": len(raw), "frames_calibrated": n_cal,
                      "frames_calibrated_pct": round(100.0 * n_cal / max(1, len(raw)), 1),
                      "global_searches": st.get("global_searches", 0), "relocks": st.get("relocks", 0),
                      "motion_failed": st.get("motion_failed", 0), "error": worker.error}
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
    hue_limit = None
    if len(cols) >= 4:
        _, labels, centers = cv2.kmeans(cols, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        # Arbitres (jaune fluo), gardiens et remplacants en chasuble : teinte tres
        # differente des deux maillots -> sans equipe (sinon ils gonflent une equipe).
        hue_d = np.linalg.norm(cols[:, 1:] - centers[labels.ravel()][:, 1:], axis=1)
        hue_limit = max(TEAM_HUE_MIN, 3.0 * float(np.median(hue_d)))
    # Numero retenu par piste : celui le plus souvent lu (vote majoritaire).
    numbers = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    def color_team(col):
        if col is None or centers is None:
            return None
        k = int(np.argmin(np.linalg.norm(centers - col, axis=1)))
        return k if np.linalg.norm(np.asarray(col)[1:] - centers[k][1:]) <= hue_limit else None

    # Equipe de chaque piste : vote sur toutes ses images (une image a l'ombre ou
    # un joueur masque ne change pas d'equipe) ; sans equipe si le plus souvent
    # hors des deux maillots (arbitre, gardien).
    team_votes = {}
    for _, pts in placed:
        for tid, x, y, ball, col in pts:
            if not ball and tid >= 0 and col is not None:
                team_votes.setdefault(tid, Counter())[color_team(col)] += 1
    track_team = {tid: c.most_common(1)[0][0] for tid, c in team_votes.items()}
    # Vue de chaque image pour les dessins tactiques cote site : homographie
    # terrain (metres) -> image (0..1), 9 nombres, H[2][2] = 1 ; None sans calage.
    def view_of(i):
        Hv = per_frame_H.get(i, (None, None))[0] if per_frame_H else None
        if Hv is not None:
            w, h = worker.size
            Hv = np.diag([1.0 / w, 1.0 / h, 1.0]) @ np.asarray(Hv, dtype=float)
        elif H is not None:
            try:
                Hv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return None
        else:
            return None
        if not np.all(np.isfinite(Hv)) or abs(Hv[2, 2]) < 1e-12:
            return None
        Hv = Hv / Hv[2, 2]
        return [float("%.6g" % v) for v in Hv.ravel()]

    tracks = []
    seen = {}
    frames = []
    for i, ((tt, pts), cal) in enumerate(zip(placed, cal_flags)):
        out = []
        for tid, x, y, ball, col in pts:
            team = track_team.get(tid) if tid >= 0 else color_team(col)
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
        frames.append({"t": tt, "points": out, "calibrated": bool(cal), "view": view_of(i)})
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
    calibration["camera_estimate"] = camera_info
    if pitch_info:
        calibration.update(pitch_info)
        calibration["calibrated"] = calibration["calibrated"] or pitch_info["frames_calibrated"] > 0
    elif camera_info.get("source") == "auto":
        calibration["method"] = "camera_tracking"
        calibration["error"] = camera_info.get("error", "camera position not found")
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
    # Trace dans les journaux RunPod (onglet Logs) : temps et calage de chaque analyse.
    print("ANALYSIS_SUMMARY " + json.dumps({"timings": timings, "pitch_calibration": calibration}), flush=True)
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


PROXY_CQ = int(os.environ.get("PROXY_CQ", "22"))   # qualite NVENC (plus petit = meilleur) ; 28 effacait les lignes lointaines


def _ffprobe_size(src):
    """(largeur, hauteur, duree) par ffprobe, pour une source que OpenCV n'ouvre pas."""
    import subprocess, json as _json
    try:
        r = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height:format=duration", "-of", "json", src],
                           capture_output=True, text=True, timeout=120)
        d = _json.loads(r.stdout or "{}")
        st = (d.get("streams") or [{}])[0]
        return int(st.get("width") or 0), int(st.get("height") or 0), float((d.get("format") or {}).get("duration") or 0) or None
    except Exception:
        return 0, 0, None


def _ffmpeg_encode(src, dst, width, fps, encoder, cq=PROXY_CQ, gpu_decode=True):
    """Re-encode la video en plus petit (largeur `width`, `fps` images/s, sans son).
    Avec gpu_decode, la video source est decodee par la puce video (NVDEC) : un
    seul flux, c'est la ou le decodage GPU fait gagner du temps (le decodage CPU
    de la video 3440 px prenait 30 a 40 min sur les machines a peu de coeurs)."""
    import subprocess
    pre = []
    if encoder == "h264_nvenc" and gpu_decode:
        # Decodage NVDEC generique (-hwaccel cuda) : marche aussi quand la source est
        # lue a distance et que ses dimensions ne sont pas connues d'avance ; le
        # redimensionnement se fait ensuite sur le processeur (bon marche).
        pre = ["-hwaccel", "cuda"]
    vf = "scale='min(%d,iw)':-2,fps=%d" % (width, fps)
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cq)]
    net = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"] if src.startswith("http") else []
    cmd = [FFMPEG, "-y", "-loglevel", "error", *net, *pre, "-i", src, "-vf", vf, "-an",
           *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr.strip()[-300:]
        if "No space left" in err:
            raise RuntimeError("%s: %s" % (encoder, err))     # inutile de reessayer autrement
        if pre:     # le decodage GPU a echoue (carte, codec) : on reessaie sur le processeur
            return _ffmpeg_encode(src, dst, width, fps, encoder, cq, gpu_decode=False)
        raise RuntimeError("%s: %s" % (encoder, err))


def prepare(inp):
    """Tache unique apres l'envoi : cree une copie allegee de la video
    (1920 de large, 15 images/s, sans son) et l'envoie a l'adresse signee
    `proxy_upload_url`. Toutes les analyses suivantes peuvent utiliser cette
    copie : telechargement, lecture et extraction bien plus rapides."""
    timings = {}
    clock = job_clock = time.time()
    # La video source est lue directement a son adresse (pas de copie locale) :
    # le disque d'un moteur RunPod (5 Go) ne tient pas l'original + la copie.
    src = inp["video_url"]
    if not src.startswith("http") or inp.get("proxy_download_source"):
        src = download(src)
        timings["download_s"] = round(time.time() - clock, 1)
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = video_end(cap)
    cap.release()
    if w <= 0:
        w, h, duration = _ffprobe_size(src)
    width = int(inp.get("proxy_width", PROXY_WIDTH))
    fps = int(inp.get("proxy_fps", PROXY_FPS))
    cq = int(inp.get("proxy_cq", PROXY_CQ))
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    clock = time.time()
    encoder, errors = None, []
    for enc in ("h264_nvenc", "libx264"):   # puce video du GPU d'abord, sinon processeur
        try:
            _ffmpeg_encode(src, dst, width, fps, enc, cq)
            encoder = enc
            break
        except Exception as e:
            errors.append(str(e)[:200])
            if "No space left" in str(e):
                break
    timings["encode_s"] = round(time.time() - clock, 1)
    local_src = not src.startswith("http")
    if encoder is None:
        if local_src:
            os.remove(src)
        if os.path.exists(dst):
            os.remove(dst)
        raise RuntimeError("proxy encoding failed: " + " | ".join(errors))
    cap = cv2.VideoCapture(dst)
    pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    size_out = os.path.getsize(dst)
    if local_src:
        size_in = os.path.getsize(src)
        os.remove(src)
    else:
        try:
            size_in = int(requests.head(src, timeout=60, allow_redirects=True).headers.get("Content-Length", 0))
        except Exception:
            size_in = None
    clock = time.time()
    with open(dst, "rb") as f:
        r = requests.put(inp["proxy_upload_url"], data=f, timeout=1800,
                         headers={"Content-Type": "video/mp4", "x-upsert": "true"})
    r.raise_for_status()
    timings["upload_s"] = round(time.time() - clock, 1)
    os.remove(dst)
    timings["total_s"] = round(time.time() - job_clock, 1)
    return {"task": "prepare", "encoder": encoder, "original": {"width": w, "height": h, "bytes": size_in},
            "proxy": {"width": pw, "height": ph, "fps": fps, "cq": cq, "bytes": size_out},
            "duration_s": round(duration, 2) if duration else None, "timings": timings}


def notify(inp, **fields):
    """Message intermediaire vers l'application (n'interrompt jamais le travail)."""
    if not inp.get("callback_url"):
        return
    try:
        body = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
        body.update(fields)
        requests.post(inp["callback_url"], json=body, timeout=60)
    except Exception:
        pass


def summarize(result):
    """Resume d'une analyse (sans les positions image par image), pour un test
    lance directement depuis la console RunPod, sans l'application."""
    # Par tranche de 5 min : part des images ou l'equipe 0 est a gauche de
    # l'equipe 1 (controle du changement de cote a la mi-temps).
    side = {}
    for f in result.get("frames", []):
        xs = {tm: [p["x"] for p in f["points"] if not p["ball"] and p["team"] == tm] for tm in (0, 1)}
        if f.get("calibrated") and len(xs[0]) >= 4 and len(xs[1]) >= 4:
            side.setdefault(int(f["t"] // 300) * 5, []).append(np.mean(xs[0]) < np.mean(xs[1]))
    timeline = {"%d min" % k: "%d%% (%d)" % (round(100 * np.mean(v)), len(v)) for k, v in sorted(side.items())}
    # Diagnostic par tranche de 5 min et par equipe : images avec >= 6 joueurs,
    # nombre median de joueurs, x median des 3 joueurs les plus a gauche / a
    # droite (gardien exclu) et x median du bloc. Sert a comparer deux analyses
    # (original / copie allegee) quand un indicateur tactique differe.
    pos = {}
    for f in result.get("frames", []):
        if not f.get("calibrated"):
            continue
        collapsed = not positions_plausible([(p["x"], p["y"]) for p in f["points"] if not p["ball"]])
        for tm in (0, 1):
            xs = sorted(p["x"] for p in f["points"] if not p["ball"] and p["team"] == tm)
            if len(xs) < 6:
                continue
            lo = xs[1:] if xs[1] - xs[0] >= KEEPER_GAP_M else xs
            hi = xs[:-1] if xs[-1] - xs[-2] >= KEEPER_GAP_M else xs
            pos.setdefault((int(f["t"] // 300) * 5, tm), []).append((len(xs), np.mean(lo[:3]), np.mean(hi[-3:]), np.mean(xs), collapsed))
    positions = {}
    for (k, tm), v in sorted(pos.items()):
        a = np.array(v)
        positions.setdefault("%d min" % k, {})["team%d" % tm] = {
            "frames": len(v), "collapsed": int(a[:, 4].sum()), "players": float(np.median(a[:, 0])),
            "left3_x": round(float(np.median(a[:, 1])), 1), "right3_x": round(float(np.median(a[:, 2])), 1),
            "mean_x": round(float(np.median(a[:, 3])), 1)}
    return {"team0_left_by_5min": timeline, "positions_by_5min": positions,
            "timings": result.get("timings"), "pitch_calibration": result.get("pitch_calibration"),
            "tactical": result.get("tactical"), "total_frames": result.get("total_frames"),
            "tracks": len(result.get("tracks", [])), "ball_frames": result.get("ball_frames"),
            "players_per_frame_median": float(np.median([len(f["points"]) for f in result.get("frames", [])] or [0]))}


def handler(job):
    inp = job["input"]
    if not inp.get("callback_url"):
        # Test direct (console RunPod) : pas d'application a prevenir, on renvoie un resume.
        if inp.get("task") == "prepare":
            try:
                return {"status": "proxy_ready", "result": prepare(inp)}
            except Exception as e:
                return {"status": "proxy_failed", "error": str(e)[:900]}
        if inp.get("task"):
            return {"status": "failed", "error": "direct test supports the default analysis and prepare only"}
        try:
            return {"status": "completed", "summary": summarize(process(inp))}
        except Exception as e:
            return {"status": "failed", "error": str(e)[:900]}
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
