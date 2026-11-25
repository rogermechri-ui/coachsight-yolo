"""Entrainement d'essai du detecteur CoachSight (joueur / gardien / arbitre).

A lancer dans un Pod RunPod avec GPU (ou n'importe quelle machine avec GPU) :

    pip install ultralytics==8.4.194
    python train_detector.py jeu-annote-lot1-mantes-argenteuil.zip

Le zip contient data.yaml, images/train/*.jpg et labels/train/*.txt (format
Ultralytics). Le script :
  1. decoupe en entrainement / validation PAR BLOCS DE TEMPS (les dernieres
     images du match restent a part : jamais de voisines quasi identiques des
     deux cotes) ; avec plusieurs matchs, mettre un match entier en validation
     avec --val-match <prefixe de nom de fichier> ;
  2. entraine yolov8m.pt (le modele actuel du moteur) sur les 3 classes ;
  3. mesure, sur la validation, le nouveau modele ET le modele actuel (qui ne
     connait que « personne ») : precision et rappel « personne », puis par
     classe pour le nouveau ;
  4. ecrit rapport.txt et best.pt dans le dossier de sortie.
"""
import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

NAMES = ["joueur", "gardien", "arbitre"]


def split_by_blocks(stems, val_fraction=0.15):
    """Les images sont triees par nom (le nom finit par l'instant en ms) ; les
    dernieres `val_fraction` vont en validation. Retourne (train, val)."""
    stems = sorted(stems)
    n_val = max(1, int(round(len(stems) * val_fraction)))
    return stems[:-n_val], stems[-n_val:]


def split_by_match(stems, val_match):
    """Un match entier (prefixe de nom) en validation."""
    val = [s for s in stems if s.startswith(val_match)]
    train = [s for s in stems if not s.startswith(val_match)]
    if not val or not train:
        raise SystemExit("--val-match %r ne donne pas un decoupage valide (%d val / %d train)" % (val_match, len(val), len(train)))
    return sorted(train), sorted(val)


def build_dataset(src, out, train, val, single=False):
    """Cree out/images/{train,val}, out/labels/{train,val}, out/data.yaml.
    single=True : toutes les classes fusionnees en « personne » (mesure du
    modele actuel, qui ne distingue pas gardien et arbitre)."""
    for part, stems in (("train", train), ("val", val)):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for s in stems:
            shutil.copy(src / "images" / "train" / (s + ".jpg"), out / "images" / part / (s + ".jpg"))
            lab = src / "labels" / "train" / (s + ".txt")
            lines = lab.read_text().splitlines() if lab.exists() else []
            if single:
                lines = ["0 " + l.split(None, 1)[1] for l in lines if l.strip()]
            (out / "labels" / part / (s + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    names = ["personne"] if single else NAMES
    (out / "data.yaml").write_text("path: %s\ntrain: images/train\nval: images/val\nnames:\n%s" % (
        out.resolve(), "".join("  %d: %s\n" % (i, n) for i, n in enumerate(names))))
    return out / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_zip")
    ap.add_argument("--base", default="yolov8m.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-match", default=None, help="prefixe des fichiers d'un match gardé en validation")
    ap.add_argument("--out", default="entrainement")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    from ultralytics import YOLO

    work = Path(a.out)
    src = work / "source"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    with zipfile.ZipFile(a.dataset_zip) as z:
        z.extractall(src)
    stems = [p.stem for p in (src / "images" / "train").glob("*.jpg")]
    if len(stems) < 20:
        raise SystemExit("trop peu d'images (%d)" % len(stems))
    train, val = split_by_match(stems, a.val_match) if a.val_match else split_by_blocks(stems)
    print("entrainement %d images, validation %d images" % (len(train), len(val)), flush=True)

    ds3 = build_dataset(src, work / "ds3", train, val)
    ds1 = build_dataset(src, work / "ds1", train, val, single=True)

    model = YOLO(a.base)
    model.train(data=str(ds3), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
                project=str(work.resolve() / "runs"), name="coachsight", exist_ok=True,
                patience=20, cos_lr=True, mosaic=1.0, close_mosaic=10,
                fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3)
    best = Path(str(model.trainer.best))  # chemin reel (Ultralytics peut ajouter runs/detect/)
    shutil.copy(best, work / "best.pt")

    lines = ["Images : %d entrainement, %d validation" % (len(train), len(val)), ""]
    new = YOLO(str(best))
    r3 = new.val(data=str(ds3), imgsz=a.imgsz, batch=a.batch, verbose=False)
    lines.append("NOUVEAU modele, 3 classes (validation) :")
    lines.append("  mAP50 toutes classes : %.3f   mAP50-95 : %.3f" % (r3.box.map50, r3.box.map))
    for i, c in enumerate(r3.box.ap_class_index):
        lines.append("  %-8s AP50 %.3f  precision %.3f  rappel %.3f" % (
            NAMES[int(c)], r3.box.ap50[i], r3.box.p[i], r3.box.r[i]))
    # « personne » : nouveau modele, classes fusionnees
    r1 = YOLO(str(best)).val(data=str(ds1), imgsz=a.imgsz, batch=a.batch, verbose=False, single_cls=True)
    lines.append("")
    lines.append("« Personne » (3 classes fusionnees), meme images :")
    lines.append("  NOUVEAU : precision %.3f  rappel %.3f  mAP50 %.3f" % (r1.box.mp, r1.box.mr, r1.box.map50))
    base = YOLO(a.base).val(data=str(ds1), imgsz=a.imgsz, batch=a.batch, verbose=False, single_cls=True, classes=[0])
    lines.append("  ACTUEL  : precision %.3f  rappel %.3f  mAP50 %.3f" % (base.box.mp, base.box.mr, base.box.map50))
    lines.append("")
    lines.append("Le modele actuel ne distingue ni gardien ni arbitre : seule la ligne « Personne » se compare.")
    (work / "rapport.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({"val_images": val}, open(work / "validation.json", "w"))


if __name__ == "__main__":
    sys.exit(main())
