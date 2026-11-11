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


# --------------------------------------------------------------------------
# Controle des adresses recues en entree (lot 2)
# --------------------------------------------------------------------------
# Le moteur telecharge, envoie et rappelle des adresses fournies dans la demande.
# Sans controle, une demande malveillante pourrait lui faire lire un service
# interne (adresse privee, metadonnees de l'hebergeur) ou un fichier local.
# Regles : https uniquement, hote public, et (si ALLOWED_URL_HOSTS est renseigne)
# hote dans la liste. URL_GUARD=off coupe le controle sans reconstruire l'image.
import ipaddress
import socket
from urllib.parse import urlsplit

FFMPEG_REMOTE_PROTOCOLS = "https,tls,tcp"
FFMPEG_LOCAL_PROTOCOLS = "file"


def _guard_on():
    return (os.environ.get("URL_GUARD") or "enforce").strip().lower() != "off"


def _allowed_hosts():
    raw = os.environ.get("ALLOWED_URL_HOSTS") or ""
    return [h.strip().lower().lstrip(".") for h in raw.split(",") if h.strip()]


def _public_ip(ip):
    a = ipaddress.ip_address(ip)
    if getattr(a, "ipv4_mapped", None):
        a = a.ipv4_mapped
    return not (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast
                or a.is_reserved or a.is_unspecified)


def url_problem(url, resolve=True):
    """None si l'adresse est acceptable, sinon la raison du refus (sans l'adresse complete)."""
    if not _guard_on():
        return None
    if not isinstance(url, str) or not url:
        return "not a text address"
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower()
        p.port   # leve ValueError si le port est invalide
    except ValueError:
        return "malformed address"
    if p.scheme != "https":
        return "only https addresses are accepted"
    if not host or p.username or p.password:
        return "malformed address"
    allowed = _allowed_hosts()
    if allowed and not any(host == h or host.endswith("." + h) for h in allowed):
        return "host %s is not in the allowed list" % host
    try:
        ips = [host] if _is_ip(host) else (
            sorted({ai[4][0] for ai in socket.getaddrinfo(host, p.port or 443, proto=socket.IPPROTO_TCP)})
            if resolve else [])
    except OSError:
        return "host %s cannot be resolved" % host
    if any(not _public_ip(ip) for ip in ips):
        return "host %s is not a public address" % host
    return None


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def check_url(url, what="address"):
    """Leve ValueError si l'adresse n'est pas acceptable."""
    why = url_problem(url)
    if why:
        raise ValueError("%s refused: %s" % (what, why))
    return url


def check_inputs(inp):
    """Controle toutes les adresses de la demande (champs *_url et *_urls).
    Renvoie None si tout va bien, sinon un message d'erreur court."""
    if not _guard_on() or not isinstance(inp, dict):
        return None
    for key, val in inp.items():
        if not isinstance(key, str) or val in (None, ""):
            continue
        if key.endswith("_url"):
            vals = [val]
        elif key.endswith("_urls"):
            vals = val if isinstance(val, (list, tuple)) else [val]
        else:
            continue
        for v in vals:
            why = url_problem(v)
            if why:
                return "input %s refused: %s" % (key, why)
    return None


def ffmpeg_protocols(src):
    """Options ffmpeg a placer avant `-i src` : la lecture ne peut suivre que les
    protocoles attendus (une video piegee ne peut pas faire ouvrir autre chose)."""
    if not _guard_on():
        return []
    remote = str(src).startswith(("http://", "https://"))
    return ["-protocol_whitelist", FFMPEG_REMOTE_PROTOCOLS if remote else FFMPEG_LOCAL_PROTOCOLS]


# --------------------------------------------------------------------------
# Limites de taille (lot 3a)
# --------------------------------------------------------------------------
# Une video ou un fichier enorme remplirait le disque du moteur et bloquerait
# les taches suivantes. Les limites se reglent par variable, sans reconstruire.

def max_video_bytes():
    """Taille maximale d'une video telechargee (MAX_VIDEO_GB, 12 Go par defaut ; 0 = sans limite)."""
    try:
        gb = float(os.environ.get("MAX_VIDEO_GB") or 12)
    except ValueError:
        gb = 12.0
    return int(gb * (1 << 30)) if gb > 0 else 0


def check_size(n, limit, what="file"):
    """Leve ValueError si `n` octets depassent `limit` (0 ou None = pas de limite)."""
    if limit and n is not None and n > limit:
        raise ValueError("%s too large: %.2f GB (max %.2f GB)" % (what, n / (1 << 30), limit / (1 << 30)))
    return n


def read_limited(resp, limit, what="file"):
    """Lit une reponse (urllib) sans jamais garder plus de `limit` octets en memoire."""
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("%s too large (max %d MB)" % (what, limit >> 20))
    return data
