"""Lot d'images a annoter (sans le modele YOLO)."""
import annotation


def test_times_are_spaced_and_inside_windows():
    wins = [[30, 330], [430, 730]]
    ts = annotation.sample_times(wins, 100, 0.37, 5.0)
    assert len(ts) == 100
    assert all(b - a >= 5.0 - 1e-6 for a, b in zip(ts, ts[1:]))
    assert all(any(ws <= t <= we for ws, we in wins) for t in ts)


def test_count_reduced_when_too_short():
    ts = annotation.sample_times([[0, 60]], 100, 0.0, 5.0)
    assert len(ts) == 12
    assert all(b - a >= 5.0 - 1e-6 for a, b in zip(ts, ts[1:]))


def test_different_phase_gives_different_times():
    a = annotation.sample_times([[0, 600]], 20, 0.1)
    b = annotation.sample_times([[0, 600]], 20, 0.6)
    assert not set(a) & set(b)


def test_empty_cases():
    assert annotation.sample_times([[0, 0]], 10, 0.5) == []
    assert annotation.sample_times([[0, 100]], 0, 0.5) == []


def test_yolo_line_normalised_and_clipped():
    assert annotation.yolo_line((100, 200, 300, 600), 1000, 1000) == "0 0.200000 0.400000 0.200000 0.400000"
    assert annotation.yolo_line((-50, 0, 100, 100), 1000, 500, 2) == "2 0.050000 0.100000 0.100000 0.200000"


def test_small_boxes_dropped_using_original_size():
    assert annotation.keep_box((0, 0, 10, 10), 0.5, 15)        # 20 px d'origine : gardee
    assert not annotation.keep_box((0, 0, 10, 6), 0.5, 15)     # 12 px d'origine : ecartee


def test_cvat_structure_files():
    train, yaml = annotation.cvat_index(["a_1", "a_2"])
    assert train == "./images/train/a_1.jpg\n./images/train/a_2.jpg\n"
    assert "train: train.txt" in yaml and "2: arbitre" in yaml


def test_detector_mode():
    class M3: names = {0: "joueur", 1: "gardien", 2: "arbitre"}
    class Coco: names = {0: "person", 1: "bicycle"}
    assert annotation.detector_mode(M3()) == (None, True)
    assert annotation.detector_mode(Coco()) == ([0], False)
