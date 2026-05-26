#!/usr/bin/env python3
"""Run the full reproducibility training flow for AI4I + PHM datasets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)]
    merged = os.environ.copy()
    deps_override = merged.get("XAI_DEPS_PATH", "").strip()
    deps_runtime = ROOT / ".deps_runtime"
    deps_legacy = ROOT / ".deps"
    if deps_override:
        deps_path = deps_override
    elif deps_runtime.exists():
        deps_path = str(deps_runtime)
    elif deps_legacy.exists():
        deps_path = str(deps_legacy)
    else:
        deps_path = ""
    prev = merged.get("PYTHONPATH", "")
    if deps_path and prev:
        merged["PYTHONPATH"] = f"{deps_path}{os.pathsep}{prev}"
    elif deps_path:
        merged["PYTHONPATH"] = deps_path
    elif prev:
        merged["PYTHONPATH"] = prev
    if env:
        merged.update(env)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=merged)


def main() -> None:
    run("prepare_phm2010_release_dataset.py")
    run("train_fast_release_models.py")
    run("train_ai4i_release_models.py")
    run("train_torch_ai4i_deep_models.py")
    run("train_phm_release_models.py")
    run("organize_trained_models.py")
    run("build_results_bundle.py")
    print("Completed full training pipeline.")


if __name__ == "__main__":
    main()
