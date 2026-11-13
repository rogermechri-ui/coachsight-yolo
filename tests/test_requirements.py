"""Les fichiers de versions du moteur doivent tout figer (lot 3b)."""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+!\-]+$")


def _lines(name):
    with open(os.path.join(ROOT, name)) as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]


@pytest.mark.parametrize("name", ["requirements.txt", "constraints.txt", "requirements-export.txt", "constraints-export.txt"])
def test_every_line_is_pinned(name):
    lines = _lines(name)
    assert lines
    bad = [l for l in lines if not PINNED.match(l)]
    assert not bad, bad


@pytest.mark.parametrize("req,cons", [("requirements.txt", "constraints.txt"), ("requirements-export.txt", "constraints-export.txt")])
def test_no_conflict_between_requirements_and_constraints(req, cons):
    c = {l.split("==")[0].lower().replace("_", "-"): l.split("==")[1] for l in _lines(cons)}
    for l in _lines(req):
        name, ver = l.split("==")
        key = name.lower().replace("_", "-")
        assert c.get(key, ver) == ver, (l, c.get(key))


def test_dockerfiles_use_the_pinned_files():
    d = open(os.path.join(ROOT, "Dockerfile")).read()
    e = open(os.path.join(ROOT, "Dockerfile.export")).read()
    assert "-c constraints.txt -r requirements.txt" in d and "COPY requirements.txt constraints.txt" in d
    assert "-c constraints-export.txt -r requirements-export.txt" in e


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.export"])
def test_engine_does_not_run_as_root(name):
    """Lot 3b, partie 2 : la derniere instruction USER designe un compte sans droits
    administrateur, placee apres les installations et avant le demarrage."""
    lines = [l.strip() for l in open(os.path.join(ROOT, name)) if l.strip() and not l.strip().startswith("#")]
    users = [(i, l.split()[1]) for i, l in enumerate(lines) if l.upper().startswith("USER ")]
    assert users, "no USER instruction"
    i, user = users[-1]
    assert user not in ("root", "0")
    assert lines[-1].startswith("CMD ") and i < len(lines) - 1
    assert not any(l.startswith(("RUN pip", "RUN apt-get", "COPY ")) for l in lines[i:])
    assert any("useradd" in l and user in l for l in lines[:i])
