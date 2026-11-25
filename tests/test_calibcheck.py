"""Controle de coherence du calage dans le temps (sans le modele YOLO)."""
import calibcheck


def _frame(t, xy, calibrated=True, ball=False):
    pts = [{"trackId": k, "x": x, "y": y, "ball": False, "team": 0} for k, (x, y) in xy.items()]
    if ball:
        pts.append({"trackId": -1, "x": 50.0, "y": 30.0, "ball": True, "team": None})
    return {"t": t, "points": pts, "calibrated": calibrated}


def _team(dx=0.0, dy=0.0):
    return {k: (20.0 + 5 * k + dx, 10.0 + 4 * k + dy) for k in range(1, 7)}


def test_smooth_play_is_kept():
    frames = [_frame(t, _team(dx=1.5 * t)) for t in range(8)]      # 1,5 m/s
    assert calibcheck.jump_frames(frames) == set()


def test_isolated_jump_is_flagged_and_next_frame_kept():
    frames = [_frame(0, _team()), _frame(1, _team(dx=1.0)), _frame(2, _team(dx=40.0)),
              _frame(3, _team(dx=3.0))]
    assert calibcheck.jump_frames(frames) == {2}


def test_lasting_shift_hidden_then_accepted_after_gap():
    # le calage change de 30 m et reste ainsi : masque quelques secondes, puis accepte
    frames = [_frame(0, _team()), _frame(1, _team())]
    frames += [_frame(t, _team(dx=30.0)) for t in range(2, 12)]
    flagged = calibcheck.jump_frames(frames)
    assert 2 in flagged and 3 in flagged
    assert 11 not in flagged                      # plus de comparaison au-dela de 5 s


def test_uncalibrated_frames_ignored_and_not_used_as_reference():
    frames = [_frame(0, _team()), _frame(1, _team(dx=500.0), calibrated=False), _frame(2, _team(dx=2.0))]
    assert calibcheck.jump_frames(frames) == set()


def test_too_few_common_players_cannot_judge():
    a = {1: (10.0, 10.0), 2: (20.0, 20.0), 3: (30.0, 30.0)}
    b = {1: (90.0, 60.0), 2: (95.0, 50.0), 3: (80.0, 40.0)}          # 3 joueurs seulement
    assert calibcheck.jump_frames([_frame(0, a), _frame(1, b)]) == set()


def test_ball_and_untracked_points_ignored():
    frames = [_frame(0, _team(), ball=True), _frame(1, _team(dx=1.0), ball=True)]
    frames[1]["points"].append({"trackId": -2, "x": 999.0, "y": 999.0, "ball": False, "team": None})
    assert calibcheck.jump_frames(frames) == set()


def test_gap_longer_than_max_resets_reference():
    frames = [_frame(0, _team()), _frame(20, _team(dx=60.0))]       # 20 s plus tard
    assert calibcheck.jump_frames(frames) == set()


def test_realistic_median_resists_a_few_fast_players():
    team = _team()
    moved = dict(team)
    for k in (1, 2):
        moved[k] = (team[k][0] + 9.0, team[k][1])                   # deux joueurs a 9 m/s
    assert calibcheck.jump_frames([_frame(0, team), _frame(1, moved)]) == set()


def test_empty_and_single():
    assert calibcheck.jump_frames([]) == set()
    assert calibcheck.jump_frames([_frame(0, _team())]) == set()
