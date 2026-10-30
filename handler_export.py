"""Petit moteur d'export sans carte graphique : ne fait que la tache
`export_clips` (decoupe et assemblage de clips avec ffmpeg). A deployer sur
un endpoint RunPod CPU (ou n'importe quel conteneur), pour qu'un telechargement
de clip ne reveille pas un worker GPU.

Meme contrat que le moteur complet :
  - mode direct (test depuis la console RunPod) : {"task": "export_clips", ...}
    sans callback_url -> renvoie le resultat ;
  - mode application : avec callback_url et callback_token -> POST du resultat
    avec status "export_ready" ou "export_failed".
Toute autre tache est refusee avec un message clair (status "unsupported").
"""
import os
import time

import requests
import runpod

import export

STARTED = time.time()


def handler(event):
    inp = event.get("input") or {}
    task = inp.get("task")
    if task != "export_clips":
        msg = "export engine: unsupported task %r (only export_clips)" % task
        if inp.get("callback_url"):
            requests.post(inp["callback_url"], json={"job_id": inp.get("job_id"), "callback_token": inp.get("callback_token"),
                                                     "status": "unsupported", "error": msg}, timeout=60)
        return {"status": "unsupported", "error": msg}
    if not inp.get("callback_url"):
        return export.export_clips(inp)
    cb = {"job_id": inp.get("job_id"), "callback_token": inp.get("callback_token"), "engine": "export-cpu"}
    try:
        cb.update(status="export_ready", result=export.export_clips(inp))
    except Exception as e:
        cb.update(status="export_failed", error=str(e)[:900])
    requests.post(inp["callback_url"], json=cb, timeout=120)
    return {"status": cb["status"]}


if __name__ == "__main__":
    print("export engine ready (cpu) · ffmpeg=%s · %s" % (export.FFMPEG, os.environ.get("RUNPOD_ENDPOINT_ID", "local")), flush=True)
    runpod.serverless.start({"handler": handler})
