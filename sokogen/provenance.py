"""Provenance stamping for result artifacts.

Operating rule 5: every experiment writes a JSON artifact containing its
config, seed, git SHA and raw counts.  Tables are generated *from* those JSONs,
never typed by hand.  This module is the single place that builds that stamp.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or ``"unknown"`` outside a repo."""
    cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
    if not short:
        cmd = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes to tracked files."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def stamp(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Build the provenance block embedded in every results JSON."""
    return {
        "git_sha": git_sha(short=False),
        "git_sha_short": git_sha(short=True),
        "git_dirty": git_dirty(),
        "seed": seed,
        "config": config,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def write_artifact(path: str, payload: Dict[str, Any]) -> None:
    """Write a results JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
