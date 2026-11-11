"""Nettoyage des messages d'erreur et des journaux avant qu'ils ne quittent le moteur.

Un message d'erreur (requests, urllib, ffmpeg) recopie souvent l'adresse appelee.
Or ces adresses portent des secrets : adresse signee du stockage (?token=...),
cle d'un service en ligne (?api_key=...). `scrub` retire ces secrets pour que
rien de sensible ne parte vers l'application ni dans les journaux RunPod.
"""
import os
import re

# Variables d'environnement dont la valeur ne doit jamais apparaitre dans un message.
SECRET_ENV = ("GEMINI_API_KEY", "ROBOFLOW_API_KEY", "RUNPOD_API_KEY", "RUNPOD_AI_API_KEY",
              "SUPABASE_SERVICE_ROLE_KEY", "CALLBACK_SECRET")

# Adresse complete : on garde l'hote et le chemin (utiles au diagnostic), on retire les parametres.
_URL_QUERY = re.compile(r"(https?://[^\s'\"<>?#]+)\?[^\s'\"<>)\]]*")
# Parametre sensible isole (adresse coupee par une troncature, texte d'ffmpeg, etc.).
_PARAM = re.compile(r"(?i)\b(token|access_token|api_key|apikey|signature|sig|x-amz-[a-z-]+|x-goog-[a-z-]+)=[^\s&'\"<>)\]]+")
# Jeton signe (JWT) : trois blocs separes par des points, commence par eyJ.
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def scrub(text, limit=None):
    """Texte sans secret : valeurs des cles connues, parametres d'adresses signees, jetons.
    `limit` : longueur maximale du resultat (coupe APRES le nettoyage)."""
    s = text if isinstance(text, str) else str(text)
    for name in SECRET_ENV:
        v = os.environ.get(name)
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    s = _URL_QUERY.sub(r"\1?***", s)
    s = _PARAM.sub(r"\1=***", s)
    s = _JWT.sub("***", s)
    return s[:limit] if limit else s
