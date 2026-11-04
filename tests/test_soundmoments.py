import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import types
# audio/clipsai importent export (ffmpeg) : on neutralise pour tester la logique pure
sys.modules.setdefault("export", types.SimpleNamespace(FFMPEG="ffmpeg"))
import soundmoments as sm


def flat(n, v=-40.0):
    return [v] * n


def test_candidates_pick_peaks_inside_windows_with_nms():
    full = flat(400)
    crowd = flat(400)
    for t in (50, 52, 200, 300):      # 50 et 52 : meme pic (NMS 6 s) ; 300 hors fenetre
        full[t] = -20.0
    c = sm.candidates({"full": full, "crowd": crowd}, 1, [(0, 250)], 10, 6)
    assert 300 not in c
    assert (50 in c) != (52 in c) or len([t for t in c if 44 <= t <= 58]) == 1
    assert 200 in c


def test_candidates_cap_and_order():
    full = flat(300)
    for t in range(20, 280, 10):
        full[t] = -30.0 + t / 100
    c = sm.candidates({"full": full, "crowd": flat(300)}, 1, [], 5, 6)
    assert len(c) == 5 and c == sorted(c)


def clip(t, events):
    return {"t": t, "ok": True, "answer": {"events": events}}


def ev(ty, tin, team="away", conf=0.9):
    return {"type": ty, "time_in_clip_s": tin, "team": team, "confidence": conf, "note": "n"}


def test_extract_filters_merges_dedups():
    clips = [clip(100, [ev("cross", 3), ev("interception", 2), ev("box_entry", 3), ev("shot", 4, conf=0.5)]),
             clip(102, [ev("cross", 1, conf=0.95)]),      # meme centre vu dans un clip voisin (t=100 : 100) -> fusion
             clip(300, [ev("free_kick", 3)]),
             clip(500, [ev("sub", 3)])]
    keep = set(sm.DEFAULT_KEEP)
    out, d = sm.extract_moments(clips, 6, keep, 0.7, [{"t": 302, "type": "free_kick"}], 6)
    types = [(m["type"], m["t"]) for m in out]
    assert d["type"] == 3          # interception, box_entry, substitution (non garde par defaut)
    assert d["confidence"] == 1    # tir 0,5
    assert d["merged"] == 1
    assert d["existing"] == 1      # coup franc deja connu
    assert [m["type"] for m in out] == ["cross"]
    assert out[0]["confidence"] == 0.95
    assert types[0][1] == 100.0


def test_team_mapping():
    assert sm._team("home", True) == "us" and sm._team("away", True) == "opp"
    assert sm._team("home", False) == "unknown"
