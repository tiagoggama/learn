"""Locate (and, when the network allows, fetch) faster-whisper model weights.

Weights are resolved in this order:

1. ``WHISPER_MODEL_DIR`` -- an explicit directory holding a CTranslate2 model.
   Use this to vendor weights that were copied into the image by hand.
2. A snapshot already present in the local Hugging Face cache (no network).
3. A fresh download from Hugging Face (needs ``huggingface.co`` to be
   reachable through the environment's network policy).
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "small"
DEFAULT_CACHE = "/opt/whisper-cache"

# A CTranslate2 model directory always carries these two files.
REQUIRED_FILES = ("model.bin", "config.json")


class ModelUnavailableError(RuntimeError):
    """Raised when weights are neither cached nor downloadable."""


def model_repo(size: str) -> str:
    """Return the Hugging Face repo id hosting the CT2 build of ``size``."""
    if size not in MODEL_SIZES:
        raise ValueError(f"unknown model size {size!r}; expected one of {', '.join(MODEL_SIZES)}")
    return f"Systran/faster-whisper-{size}"


def cache_dir() -> Path:
    """Directory used as ``HF_HOME``. Lives outside the repo so it survives re-clones."""
    return Path(os.environ.get("WHISPER_CACHE", DEFAULT_CACHE))


def is_model_dir(path: Path) -> bool:
    """True if ``path`` looks like a usable CTranslate2 model directory."""
    return path.is_dir() and all((path / name).exists() for name in REQUIRED_FILES)


def vendored_model_dir() -> Path | None:
    """Return the operator-supplied model directory, if it is set and usable."""
    raw = os.environ.get("WHISPER_MODEL_DIR")
    if not raw:
        return None
    path = Path(raw)
    return path if is_model_dir(path) else None


def ensure_model(size: str = DEFAULT_MODEL, *, allow_download: bool = True) -> Path:
    """Return a local path to the weights for ``size``.

    Raises :class:`ModelUnavailableError` with remediation steps when the
    weights are missing and cannot be fetched.
    """
    vendored = vendored_model_dir()
    if vendored is not None:
        return vendored

    # Import lazily so that importing this module never requires the dependency.
    from huggingface_hub import snapshot_download

    repo = model_repo(size)
    os.environ.setdefault("HF_HOME", str(cache_dir()))

    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except Exception:  # not cached yet -- fall through to the network attempt
        pass

    if not allow_download:
        raise ModelUnavailableError(_missing_message(repo))

    try:
        return Path(snapshot_download(repo))
    except Exception as exc:
        raise ModelUnavailableError(f"{_missing_message(repo)}\n\nUnderlying error: {exc}") from exc


def _missing_message(repo: str) -> str:
    return (
        f"Whisper weights for {repo} are not available offline.\n"
        "Fix it in one of two ways:\n"
        f"  1. Allow 'huggingface.co' in this environment's network policy, then rerun "
        f"the SessionStart hook so the weights are cached into the container image.\n"
        f"  2. Copy a CTranslate2 model directory into the image and point "
        f"WHISPER_MODEL_DIR at it (it must contain {', '.join(REQUIRED_FILES)})."
    )
