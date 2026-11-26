"""Tache `annotation_batch` : lot d'images a annoter a la main pour entrainer
notre propre detecteur (etape 2 : classes joueur, gardien, arbitre).

Entree (input) :
  video_url        lien de la video du match
  count            nombre d'images (defaut 100)
  min_gap_s        ecart minimal entre deux images (defaut 5 s)
  phase            decalage 0..1 des instants (defaut au hasard : deux lots sur
                   le meme match ne tombent jamais sur les memes images)
  play_windows     periodes de jeu [[debut, fin], ...] (secondes)
  match_name       nom du match, ecrit dans le manifeste
  conf             confiance minimale de la pre-annotation (defaut 0.25)
  min_box_px       hauteur minimale d'une boite, en pixels de l'image d'origine
                   (defaut 15, regle du guide d'annotation)
  max_width        largeur maximale des images (defaut 1920)
  upload_url       adresse signee ou deposer le zip (PUT)
Sortie : zip au format Ultralytics tel que CVAT l'importe (images/train/, labels/train/,
train.txt, classes.txt, data.yaml)
plus manifest.json. Avec le modele d'origine, les boites pre-annotees sont toutes de classe 0 (« joueur ») ;
avec le detecteur CoachSight a 3 classes (ANNOTATION_MODEL), elles portent joueur / gardien / arbitre. Dans tous les cas :
c'est a la personne qui annote de supprimer les fausses boites (staff, public,
remplacants) et de passer les gardiens (classe 1) et arbitres (classe 2).
"""
import io
import json
import os
import random
import zipfile

import cv2
import requests

CLASSES = ["joueur", "gardien", "arbitre"]
MIN_GAP_S = 5.0


def sample_times(windows, count, phase, min_gap_s=MIN_GAP_S):
    """Instants regulierement espaces sur le temps de jeu (jamais deux images a
    moins de `min_gap_s` secondes l'une de l'autre). Le nombre d'images est
    reduit si le temps de jeu est trop court pour respecter l'ecart."""
    total = sum(max(0.0, we - ws) for ws, we in windows)
    if total <= 0 or count <= 0:
        return []
    count = min(int(count), max(1, int(total // max(min_gap_s, 0.001))))
    times = []
    for k in range(count):
        pos = (k + phase) * total / count
        for ws, we in windows:
            if pos <= we - ws:
                times.append(round(ws + pos, 2))
                break
            pos -= we - ws
    return times


def yolo_line(box, width, height, cls=0):
    """Ligne au format YOLO : classe, centre x, centre y, largeur, hauteur (0..1)."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    return "%d %.6f %.6f %.6f %.6f" % (cls, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                                       (x2 - x1) / width, (y2 - y1) / height)


def cvat_index(names):
    """Fichiers de structure attendus par l'import « Ultralytics YOLO Detection » de
    CVAT : images/train/, labels/train/, train.txt (liste des images) et data.yaml."""
    train = "".join("./images/train/%s.jpg\n" % n for n in names)
    yaml = "path: .\ntrain: train.txt\nnames:\n" + "".join("  %d: %s\n" % (i, c) for i, c in enumerate(CLASSES))
    return train, yaml


def keep_box(box, scale, min_box_px):
    """Vrai si la boite fait au moins `min_box_px` de haut dans l'image d'origine
    (`scale` = taille de l'image enregistree / taille d'origine)."""
    return (box[3] - box[1]) / max(scale, 1e-6) >= min_box_px


def detector_mode(model):
    """Modele a 3 classes (joueur, gardien, arbitre) : on garde ses classes.
    Sinon (modele COCO d'origine) : « personne » (classe 0), tout est pre-annote joueur."""
    names = getattr(model, "names", None) or {}
    if [names[k] for k in sorted(names)] == CLASSES:
        return None, True
    return [0], False


def annotation_batch(inp, model, download, imgsz, play_windows, video_end, scrub):
    only_classes, own_classes = detector_mode(model)
    count = int(inp.get("count", 100))
    min_gap = float(inp.get("min_gap_s", MIN_GAP_S))
    conf = float(inp.get("conf", 0.25))
    min_px = float(inp.get("min_box_px", 15))
    max_w = int(inp.get("max_width", 1920))
    phase = float(inp["phase"]) if inp.get("phase") is not None else random.random()
    src = download(inp["video_url"])
    cap = cv2.VideoCapture(src)
    try:
        dur = video_end(cap) or 0
        windows = play_windows(inp, 0.0, dur) if dur else [[0.0, 0.0]]
        times = sample_times(windows, count, phase, min_gap)
        buf = io.BytesIO()
        zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED)
        items, names, boxes_total = [], [], 0
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h0, w0 = frame.shape[:2]
            # agnostic_nms : une personne = un seul cadre (sinon v2 dessine un cadre joueur ET un cadre gardien/arbitre)
            res = model(frame, classes=only_classes, conf=conf, imgsz=imgsz, verbose=False, agnostic_nms=own_classes)[0]
            if w0 > max_w:
                frame = cv2.resize(frame, (max_w, int(h0 * max_w / w0)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            scale = w / w0
            lines, boxes = [], []
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) * scale for v in b.xyxy[0].tolist()]
                if not keep_box((x1, y1, x2, y2), scale, min_px):
                    continue
                lines.append(yolo_line((x1, y1, x2, y2), w, h, int(b.cls[0]) if own_classes else 0))
                boxes.append(round(float(b.conf[0]), 3))
            name = "%s_%09d" % (inp.get("match_slug", "match"), int(round(t * 1000)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            zf.writestr("images/train/%s.jpg" % name, jpg.tobytes())
            zf.writestr("labels/train/%s.txt" % name, "\n".join(lines) + ("\n" if lines else ""))
            names.append(name)
            items.append({"t": t, "file": "images/train/%s.jpg" % name, "width": w, "height": h,
                          "boxes": len(lines), "confidences": boxes})
            boxes_total += len(lines)
    finally:
        cap.release()
        os.remove(src)
    zf.writestr("classes.txt", "\n".join(CLASSES) + "\n")
    train_txt, data_yaml = cvat_index(names)
    zf.writestr("train.txt", train_txt)
    zf.writestr("data.yaml", data_yaml)
    zf.writestr("manifest.json", json.dumps({
        "match": inp.get("match_name"), "classes": CLASSES, "phase": round(phase, 3),
        "min_gap_s": min_gap,
        "pre_annotation": ("detecteur CoachSight 3 classes : verifier et corriger a la main" if own_classes
                           else "tous « joueur » : corriger a la main"),
        "items": items}))
    zf.close()
    data = buf.getvalue()
    r = requests.put(inp["upload_url"], data=data, timeout=600,
                     headers={"Content-Type": "application/zip", "x-upsert": "true"})
    r.raise_for_status()
    return {"task": "annotation_batch", "images": len(items), "boxes": boxes_total,
            "bytes": len(data), "phase": round(phase, 3)}
