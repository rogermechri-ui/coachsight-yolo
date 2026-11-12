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
