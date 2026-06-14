"""Python 3.8.3 compatibility guards for the locked-down target environment.

* every module byte-compiles, and
* no Python 3.9+ only idioms slip in.

This does not *run* under 3.8 here (the dev box is newer), but it catches the
common syntax/stdlib regressions that would break on 3.8.
"""

import os
import py_compile
import re

import ruptures_lite

PKG_DIR = os.path.dirname(ruptures_lite.__file__)


def _all_modules():
    for root, _dirs, files in os.walk(PKG_DIR):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def test_all_modules_compile():
    for path in _all_modules():
        py_compile.compile(path, doraise=True)


# Patterns that exist only in Python >= 3.9 (PEP 585 generics appear only inside
# annotations, which are safe thanks to `from __future__ import annotations`, so
# they are intentionally not flagged here).
BANNED = [
    (r"\.removeprefix\(", "str.removeprefix (3.9+)"),
    (r"\.removesuffix\(", "str.removesuffix (3.9+)"),
    (r"\bmath\.lcm\b", "math.lcm (3.9+)"),
    (r"^\s*match\s+.+:\s*$", "match statement (3.10+)"),
    (r"\bgraphlib\b", "graphlib (3.9+)"),
]


def test_no_py39_idioms():
    compiled = [(re.compile(pat, re.MULTILINE), msg) for pat, msg in BANNED]
    offenders = []
    for path in _all_modules():
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for rx, msg in compiled:
            if rx.search(src):
                offenders.append("{}: {}".format(os.path.relpath(path, PKG_DIR), msg))
    assert not offenders, "Found 3.9+ idioms:\n" + "\n".join(offenders)


def test_future_annotations_present():
    """Modules using annotations should import `annotations` from __future__."""
    missing = []
    for path in _all_modules():
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        if "from __future__ import annotations" not in src:
            # only flag files that actually use `->` or `: <type>` annotations
            if "->" in src or re.search(r"def \w+\([^)]*:\s*\w", src):
                missing.append(os.path.relpath(path, PKG_DIR))
    assert not missing, "Missing future-annotations import:\n" + "\n".join(missing)
