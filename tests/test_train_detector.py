"""Decoupage et jeux de donnees de l'entrainement d'essai (sans ultralytics)."""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("train_detector", Path(__file__).resolve().parent.parent / "tools" / "train_detector.py")
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)


def test_block_split_keeps_latest_images_for_validation():
    stems = ["m_%09d" % t for t in range(0, 100000, 1000)]
    train, val = td.split_by_blocks(list(reversed(stems)), 0.15)
    assert len(val) == 15 and len(train) == 85
    assert max(train) < min(val)


def test_match_split_and_errors():
    stems = ["a_1", "a_2", "b_1", "b_2", "b_3"]
    train, val = td.split_by_match(stems, "b_")
    assert val == ["b_1", "b_2", "b_3"] and train == ["a_1", "a_2"]
    with pytest.raises(SystemExit):
        td.split_by_match(stems, "zzz")


def _src(tmp_path):
    src = tmp_path / "src"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    for s, lab in (("a", "0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.1 0.2\n"), ("b", "2 0.3 0.3 0.1 0.2\n"), ("c", "")):
        (src / "images" / "train" / (s + ".jpg")).write_bytes(b"x")
        (src / "labels" / "train" / (s + ".txt")).write_text(lab)
    return src


def test_dataset_three_classes(tmp_path):
    out = tmp_path / "d3"
    y = td.build_dataset(_src(tmp_path), out, ["a", "b"], ["c"])
    assert "2: arbitre" in y.read_text()
    assert (out / "labels" / "train" / "b.txt").read_text().startswith("2 ")
    assert (out / "labels" / "val" / "c.txt").read_text() == ""


def test_dataset_single_class_merges_everything(tmp_path):
    out = tmp_path / "d1"
    y = td.build_dataset(_src(tmp_path), out, ["a", "b"], ["c"], single=True)
    assert "0: personne" in y.read_text()
    assert (out / "labels" / "train" / "a.txt").read_text().splitlines() == ["0 0.5 0.5 0.1 0.2", "0 0.2 0.2 0.1 0.2"]
    assert (out / "labels" / "train" / "b.txt").read_text().startswith("0 ")
