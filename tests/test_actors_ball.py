import numpy as np
import actors


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_ball_marker_drawn_when_seen():
    img = actors._draw_context(_frame(), [], 1280, ball=(640, 360))
    # anneau jaune (BGR 0,230,255) autour du ballon, rien ailleurs
    assert (img[:, :, 2] == 255).any()
    assert not (_frame() == img).all()


def test_no_marker_without_ball():
    img = actors._draw_context(_frame(), [], 1280, ball=None)
    assert (img == _frame()).all()


def test_marker_scales_with_context_width():
    img = actors._draw_context(_frame(), [], 640, ball=(640, 360))
    assert img.shape[1] == 640
    # le ballon est redessine a (320, 180) dans l'image reduite
    assert img[170:190, 300:340].any()
