# Pitch calibration with a fixed Veo camera (research, not used by the engine)

Veo follow-cam views are virtual crops of one fixed panoramic camera, so each frame
is defined by the camera position C (once per match) plus pan, tilt and focal length.

- `mask2.py`: pitch area (grass below the boards) and thin white-line mask (adaptive threshold, perspective-aware thickness).
- `cam.py`: camera model `homog(C, pan, tilt, f)`, two-way line cost (drawn lines -> white pixels, white pixels -> pitch lines in metres), global grid search `solve2`.
- `track.py` / `track2.py`: frame-to-frame tracking; `track2` predicts camera motion from background features (ORB + homography) and refines on lines.
- `calib.py`, `run.py`, `show.py`: experiments and overlays.

Status (2026-10-04, match vs Emory & Henry, Frock Field 105 x 68):
- Frame with the centre circle: camera fit OK (C ~ (52.4, 68.5, 2.9) m; to be confirmed).
- Global search on single frames: OK on mid shots, fails on wide shots (far lines 1 px).
- Tracking with motion prediction keeps the circle roughly in place over 2 min, but far
  lines drift (camera position / lens model not yet exact).

Next: estimate C jointly over many tracked frames; check lens model (principal point,
distortion); re-anchor on the centre circle / penalty boxes when visible.
