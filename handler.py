"""CoachSight AI - moteur de detection YOLO pour RunPod Serverless.
Recoit la video d'un match, detecte joueurs et ballon, renvoie les positions
a l'application (callback_url)."""
import os, tempfile, requests, cv2, numpy as np, runpod
from ultralytics import YOLO

MODEL = YOLO(os.environ.get("YOLO_MODEL", "yolov8m.pt"))
PITCH_L, PITCH_W = 105.0, 68.0
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2"))

def shirt_color(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    crop = frame[max(y1 + int(h * 0.15), 0):y1 + int(h * 0.5), max(x1, 0):x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)

def process(inp):
    url, start, end = inp["video_url"], float(inp.get("start_s", 0)), float(inp["end_s"])
    fps_out = min(float(inp.get("fps", SAMPLE_FPS)), SAMPLE_FPS)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    cap = cv2.VideoCapture(tmp)
    raw = []  # (t, [(trackId, x, y, ball, color)])
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = MODEL.track(frame, persist=True, classes=[0, 32], conf=0.25, verbose=False)[0]
        pts = []
        if res.boxes is not None:
            ids = res.boxes.id.tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for box, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), ids):
                ball = int(cls) == 32
                cx, cy = (box[0] + box[2]) / 2, box[3] if not ball else (box[1] + box[3]) / 2
                col = None if ball else shirt_color(frame, box)
                pts.append((int(tid) if tid is not None else -1, cx / w * PITCH_L, cy / h * PITCH_W, ball, col))
        raw.append((round(t, 2), pts))
        t += 1.0 / fps_out
    cap.release(); os.remove(tmp)

    cols = np.array([p[4] for _, pts in raw for p in pts if p[4] is not None], dtype=np.float32)
    centers = None
    if len(cols) >= 4:
        _, _, centers = cv2.kmeans(cols, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    frames = []
    for tt, pts in raw:
        out = []
        for tid, x, y, ball, col in pts:
            team = None
            if col is not None and centers is not None:
                team = int(np.argmin(np.linalg.norm(centers - col, axis=1)))
            out.append({"trackId": tid, "team": team, "x": round(x, 2), "y": round(y, 2), "ball": ball})
        frames.append({"t": tt, "points": out})
    return {"simulated": False, "pitch": {"length": PITCH_L, "width": PITCH_W}, "frames": frames}

def handler(job):
    inp = job["input"]
    cb = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
    try:
        result = process(inp)
        cb.update(status="completed", result=result, stats={"frames": len(result["frames"])})
    except Exception as e:
        cb.update(status="failed", error=str(e)[:900])
    requests.post(inp["callback_url"], json=cb, timeout=120)
    return {"status": cb["status"], "frames": len(cb.get("result", {}).get("frames", []))}

runpod.serverless.start({"handler": handler})
