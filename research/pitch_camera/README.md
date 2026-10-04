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

Update (2026-10-04 evening):
- `calib_pts.py`: camera position from hand-measured features in 3 views (centre circle,
  both goals' posts, near touchline): C = (52.37, 71.47, 3.14) m, pinhole model, RMS 0.9 px.
  So the Veo view is a plain pinhole (no visible distortion), camera 3.5 m behind the near
  touchline at halfway, 3.1 m high.
- `track2.py` with this C: pitch correctly tracked over most of a 2-min clip (fast pans,
  corners, near touchline), using background motion (ORB) + line refinement with an
  adaptive window. ~0.45 s/frame on 1 CPU (to optimise).
- Global search (`solve2`) works on near/mid shots, still fails on very wide shots: use it
  only to (re)initialise on easy frames, then track.
