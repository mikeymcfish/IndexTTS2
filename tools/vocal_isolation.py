"""Utility helpers for isolating vocals from a mixed audio track.

This module wraps the Demucs command line interface used by the
ComfyUI-DeepExtractV2 workflow referenced by the user.  The helper keeps the
heavy lifting in a dedicated function so the Gradio UI can stay lean while
still offering a straightforward way to trigger the extraction process.
"""

from __future__ import annotations

import glob
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional


class VocalIsolationError(RuntimeError):
    """Raised when the vocal isolation command fails."""


@dataclass
class VocalIsolationResult:
    """Structured response returned after the isolation finishes."""

    vocals_path: str
    accompaniment_path: Optional[str]
    logs: str
    workspace_dir: str


def _resolve_two_stem_artifacts(folder: str) -> tuple[Optional[str], Optional[str]]:
    """Locate generated audio artifacts inside the Demucs output tree."""

    vocals = None
    accompaniment = None

    for candidate in glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True):
        lower_name = os.path.basename(candidate).lower()
        if "vocals" in lower_name and vocals is None:
            vocals = candidate
        elif ("no_vocals" in lower_name or "instrumental" in lower_name) and accompaniment is None:
            accompaniment = candidate

    return vocals, accompaniment


def isolate_vocals_with_demucs(
    audio_path: str,
    *,
    model_name: str = "htdemucs",
    device: Optional[str] = None,
    segment_length: Optional[float] = None,
    shifts: Optional[int] = None,
    overlap: Optional[float] = None,
) -> VocalIsolationResult:
    """Run Demucs in two-stem mode (vocals/no_vocals) and return the result.

    Parameters
    ----------
    audio_path:
        Absolute path to the input audio file.
    model_name:
        Demucs model to use.  Matches the ``-n`` argument of the CLI.
    device:
        Target device (``"cuda"``, ``"cpu"`` or ``None`` to let Demucs decide).
    segment_length:
        Optional chunk size passed through to ``--segment``.
    shifts:
        Optional number of shifts (``--shifts``) to average for higher quality.
    overlap:
        Optional overlap ratio forwarded to ``--overlap``.

    Returns
    -------
    VocalIsolationResult
        Paths to the generated stems and the captured Demucs logs.

    Raises
    ------
    ModuleNotFoundError
        If the ``demucs`` package is not available in the current environment.
    VocalIsolationError
        If the separation command exits with a non-zero return code or no
        expected artifacts are produced.
    """

    try:
        importlib.import_module("demucs")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'demucs' package is required for vocal isolation. Install it with "
            "`pip install demucs` before using this feature."
        ) from exc

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    tmp_output = tempfile.mkdtemp(prefix="deepextract_")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        model_name,
        "-o",
        tmp_output,
        audio_path,
        "--jobs",
        "1",
    ]

    if device:
        cmd.extend(["-d", device])

    if segment_length and segment_length > 0:
        cmd.extend(["--segment", str(segment_length)])

    if shifts and shifts > 0:
        cmd.extend(["--shifts", str(int(shifts))])

    if overlap is not None:
        cmd.extend(["--overlap", str(overlap)])

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    logs = process.stdout or ""

    if process.returncode != 0:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise VocalIsolationError(
            f"Demucs exited with code {process.returncode}. Command: {' '.join(cmd)}\n{logs}"
        )

    vocals, accompaniment = _resolve_two_stem_artifacts(tmp_output)

    if not vocals:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise VocalIsolationError(
            "Demucs finished without producing a vocals stem. Check the logs for details."
        )

    return VocalIsolationResult(
        vocals_path=vocals,
        accompaniment_path=accompaniment,
        logs=logs,
        workspace_dir=tmp_output,
    )


def cleanup_paths(paths: Iterable[Optional[str]]) -> None:
    """Remove temporary files or folders created during extraction."""

    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)

