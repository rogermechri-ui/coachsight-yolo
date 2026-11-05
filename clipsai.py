"""Lecture de courts clips video par Gemini : tache `clips_ai` du moteur d'export (sans GPU).

Pourquoi : sur une image fixe de plan large, la plupart des actions rapides sont
invisibles. Un clip de quelques secondes en mouvement les montre. Test (pilote) :
couper un clip autour de chaque « moment candidat » (pics sonores) et demander
a Gemini ce qui s'y passe.

Entree :
  video_url        adresse lisible (signee) de la video du match
  times            liste d'instants (secondes) = centre de chaque clip
  clip_s           duree d'un clip (defaut 6)
  model            ex. "gemini-3.6-flash" (obligatoire, aucun defaut cache)
  thinking_budget  jetons de reflexion (defaut 0)
  home_colors, away_colors   facultatif, ex. "white with blue", pour l'equipe
  max_clips        garde-fou (defaut 80)
  workers          appels paralleles (defaut 4)
La cle est lue dans la variable d'environnement GEMINI_API_KEY (jamais dans l'entree,
jamais dans un journal ni dans le resultat).
Sortie : {task, model, n, clips:[{t, ok, answer|error, in_tok, out_tok}], usage{in_tok,out_tok,
  est_cost_usd}, timings}
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from export import FFMPEG

API = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
# prix indicatifs en $ par million de jetons (entree, sortie) : a ajuster si le tarif change
PRICES = {"flash": (0.75, 3.75), "flash-lite": (0.30, 2.50), "pro": (2.0, 12.0)}

TYPES = ["goal", "shot", "big_chance", "save", "cross", "corner", "free_kick", "penalty", "foul",
         "duel", "tackle", "interception", "turnover", "counter_attack", "box_entry", "injury",
         "substitution", "none"]

PROMPT = """You are a football (soccer) match analyst. The video is a %ds clip from a wide TV broadcast (players are small). \
List the notable actions that happen in the clip, if any. %s
Action types and how to recognise them:
- goal: the ball enters the net, celebration follows.
- shot: a player strikes the ball towards goal (on or off target, blocked).
- save: the goalkeeper stops or deflects a shot.
- big_chance: a clear scoring opportunity that is not converted (one-on-one, open goal, header from close range).
- cross: a ball played from a wide area (near the touchline) into or towards the penalty area, usually in the air.
- corner: the ball is played from the corner arc.
- free_kick: play restarts with a kick from a fixed spot after a stoppage (players stand still, a player prepares to kick a ball placed on the ground).
- penalty: a kick from the penalty spot.
- foul: a clear contact or trip that stops play or makes a player fall; the referee may react.
- duel: a close one-on-one contest for the ball (no foul).
- aerial_duel: two players jump for a ball in the air.
- tackle: a player slides or steps in to take the ball from an opponent.
- interception: a player cuts out a pass.
- turnover: a team clearly loses possession (bad pass, lost control).
- counter_attack: a fast attack straight after winning the ball, many players running forward.
- box_entry: the ball or a ball carrier enters the penalty area.
- injury: a player stays down and receives treatment.
- sub: substitution (players leaving or entering at the touchline, board held up).
Do not list normal passing or build-up play. If nothing notable happens, return an empty list.
Answer with JSON only: {"events":[{"type":"<one type>","time_in_clip_s":<seconds from clip start, number>,"team":"home"|"away"|"unknown","confidence":<0..1>,"note":"<max 15 words>"}]}."""


def _price(model):
    m = model.lower()
    k = "flash-lite" if "lite" in m else ("pro" if "pro" in m else "flash")
    return PRICES[k]


def _cut(src, t, clip_s, out):
    start = max(0.0, t - clip_s / 2.0)
    net = (["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
           if str(src).startswith("http") else [])
    args = [FFMPEG, "-y", "-loglevel", "error", "-nostdin", *net, "-ss", "%.2f" % start, "-i", src,
            "-t", str(clip_s), "-an", "-vf", "scale=-2:480,fps=10",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-movflags", "+faststart", out]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("ffmpeg clip failed: " + (r.stderr or "")[-300:])


def _ask(key, model, mp4, prompt, thinking_budget):
    body = {"contents": [{"parts": [
                {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(mp4).decode()}},
                {"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1,
                                 "thinkingConfig": {"thinkingBudget": int(thinking_budget)}}}
    req = urllib.request.Request(API % model, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = "HTTP %d %s" % (e.code, (e.read() or b"")[:200].decode("utf-8", "ignore").replace(key, "***"))
            if e.code in (429, 500, 503):
                time.sleep(10 * (attempt + 1))
                continue
            break
        except Exception as e:  # reseau
            last = str(e).replace(key, "***")[:200]
            time.sleep(5)
    raise RuntimeError(last or "gemini call failed")


def clips_ai(inp, cutter=None):
    t0 = time.time()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set on this endpoint")
    model = inp.get("model")
    if not model:
        raise RuntimeError("model is required")
    src = None if cutter else inp["video_url"]
    clip_s = int(inp.get("clip_s") or 6)
    times = [float(x) for x in (inp.get("times") or [])]
    cap = int(inp.get("max_clips") or 80)
    if not times:
        raise RuntimeError("times is empty")
    if len(times) > cap:
        raise RuntimeError("too many clips: %d (max_clips %d)" % (len(times), cap))
    colors = ""
    if inp.get("home_colors") or inp.get("away_colors"):
        colors = 'Teams: "home" wears %s, "away" wears %s.' % (inp.get("home_colors") or "?", inp.get("away_colors") or "?")
    prompt = PROMPT % (clip_s, colors)
    work = tempfile.mkdtemp(prefix="clipsai_")
    budget = inp.get("thinking_budget", 0)

    deadline = t0 + float(inp.get("deadline_s") or 1800)
    done = [0]

    def one(t):
        out = os.path.join(work, "c_%d.mp4" % int(t * 10))
        if time.time() > deadline:
            return {"t": t, "ok": False, "error": "deadline"}
        try:
            (cutter or (lambda tt, cs, o: _cut(src, tt, cs, o)))(t, clip_s, out)
            mp4 = open(out, "rb").read()
            resp = _ask(key, model, mp4, prompt, budget)
            u = resp.get("usageMetadata") or {}
            txt = ""
            for c in resp.get("candidates") or []:
                for p in (c.get("content") or {}).get("parts") or []:
                    txt += p.get("text") or ""
            try:
                ans = json.loads(txt)
            except ValueError:
                ans = {"raw": txt[:300]}
            return {"t": t, "ok": True, "answer": ans, "in_tok": u.get("promptTokenCount", 0),
                    "out_tok": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0),
                    "mp4_kb": round(len(mp4) / 1024)}
        except Exception as e:
            return {"t": t, "ok": False, "error": str(e).replace(key, "***")[:300]}
        finally:
            done[0] += 1
            if done[0] % 10 == 0:
                print("clips_ai progress %d/%d after %ds" % (done[0], len(times), time.time() - t0), flush=True)
            try:
                os.remove(out)
            except OSError:
                pass

    with ThreadPoolExecutor(max_workers=int(inp.get("workers") or 4)) as ex:
        res = list(ex.map(one, times))
    try:
        os.rmdir(work)
    except OSError:
        pass
    ti = sum(r.get("in_tok", 0) for r in res)
    to = sum(r.get("out_tok", 0) for r in res)
    pi, po = _price(model)
    return {"task": "clips_ai", "model": model, "n": len(res), "failed": sum(1 for r in res if not r["ok"]),
            "clips": res,
            "usage": {"in_tok": ti, "out_tok": to, "est_cost_usd": round(ti / 1e6 * pi + to / 1e6 * po, 4)},
            "timings": {"total_s": round(time.time() - t0, 1)}}
