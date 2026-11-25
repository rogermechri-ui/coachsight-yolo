"""Filtre pelouse et equipes par couleurs du coach (sans le modele YOLO)."""
import cv2
import numpy as np

import coachteams
import pitchmask


def _scene(w=640, h=360, grass_bgr=(60, 160, 60)):
    img = np.full((h, w, 3), (90, 90, 90), np.uint8)       # tribune grise en haut
    img[int(h * 0.35):] = grass_bgr                          # pelouse en bas
    return img


def test_grass_mask_and_feet():
    img = _scene()
    mask, hull = pitchmask.grass_mask(img)
    assert hull is not None
    assert pitchmask.foot_on_pitch(mask, hull, [300, 200, 320, 260], (640, 360))      # sur la pelouse
    assert not pitchmask.foot_on_pitch(mask, hull, [300, 40, 320, 90], (640, 360))    # dans la tribune


def test_grass_yellowish_still_found():
    img = _scene(grass_bgr=(50, 170, 150))   # pelouse jaune-vert
    mask, hull = pitchmask.grass_mask(img)
    assert hull is not None


def test_no_grass_means_no_filter():
    img = np.full((360, 640, 3), 120, np.uint8)
    assert pitchmask.grass_mask(img) == (None, None)


def test_off_pitch_tracks_and_safeguard():
    obs = {1: [True] * 5, 2: [False] * 5, 3: [True, False, True, True], 4: [False, False]}
    assert pitchmask.off_pitch_tracks(obs) == {2}        # 4 : trop peu de vues
    # tout serait retire : pelouse mal reperee, on ne retire rien
    assert pitchmask.off_pitch_tracks({1: [False] * 5, 2: [False] * 5}) == set()


def _hex_lab_obs(hx, n=6, jitter=2.0, seed=0, dull=0.8):
    rng = np.random.default_rng(seed)
    lab = coachteams.hex_to_lab(hx)
    lab = np.array([lab[0] * dull, 128 + (lab[1] - 128) * dull, 128 + (lab[2] - 128) * dull], np.float32)
    return [lab + rng.normal(0, jitter, 3).astype(np.float32) for _ in range(n)]


def _placed(groups):
    placed, tid = [], 0
    frames = {}
    for hx, count, seed in groups:
        for k in range(count):
            for i, c in enumerate(_hex_lab_obs(hx, seed=seed * 100 + k)):
                frames.setdefault(i, []).append((tid, 0.5, 0.5, False, np.append(c, np.nan), None))
            tid += 1
    for i, pts in sorted(frames.items()):
        placed.append((float(i), pts))
    return placed


def test_teams_from_coach_colors_assigns_by_seed():
    placed = _placed([("#c81e1e", 8, 1), ("#1e3cc8", 8, 2)])
    out = coachteams.teams_from_coach_colors(placed, "#1e3cc8", "#c81e1e")
    assert out is not None
    track_team, _fn, rgb = out
    assert all(track_team[t] == 1 for t in range(8))           # rouges = adversaire (1)
    assert all(track_team[t] == 0 for t in range(8, 16))       # bleus = coach (0)
    assert rgb.shape == (2, 3)


def test_teams_fallbacks():
    placed = _placed([("#c81e1e", 8, 1), ("#1e3cc8", 8, 2)])
    assert coachteams.teams_from_coach_colors(placed, None, "#c81e1e") is None
    assert coachteams.teams_from_coach_colors(placed, "zzz", "#c81e1e") is None
    assert coachteams.teams_from_coach_colors(placed, "#c81e1e", "#c82020") is None   # trop proches
    assert coachteams.teams_from_coach_colors(placed[:1], "#1e3cc8", "#c81e1e") is None   # trop peu de joueurs


def test_referee_without_team():
    placed = _placed([("#c81e1e", 8, 1), ("#1e3cc8", 8, 2), ("#e6d21e", 1, 3)])
    track_team, _fn, _rgb = coachteams.teams_from_coach_colors(placed, "#1e3cc8", "#c81e1e")
    assert 16 not in track_team           # le jaune (arbitre) n'a pas d'equipe
