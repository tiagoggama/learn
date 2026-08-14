"""Locate (and, when the network allows, fetch) faster-whisper model weights.

Weights are resolved in this order:

1. ``WHISPER_MODEL_DIR`` -- an explicit directory holding a CTranslate2 model.
2. ``models/faster-whisper-<size>/`` inside the repo -- weights committed to
   git. This is the route that works when ``huggingface.co`` is blocked, since
   the repo is cloned over an allowed git transport.
3. A snapshot already present in the local Hugging Face cache (no network).
4. A fresh download from Hugging Face (needs ``huggingface.co`` to be
   reachable through the environment's network policy).
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "small"
DEFAULT_CACHE = "/opt/whisper-cache"
VENDOR_SUBDIR = "models"

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


LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an unfetched Git LFS pointer rather than real content.

    Cloning a repo without ``git lfs pull`` leaves tiny text stand-ins where the
    weights should be, which CTranslate2 reports only as an opaque load failure.
    """
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False


def is_model_dir(path: Path) -> bool:
    """True if ``path`` looks like a usable CTranslate2 model directory."""
    if not path.is_dir() or not all((path / name).exists() for name in REQUIRED_FILES):
        return False
    return not is_lfs_pointer(path / "model.bin")


def repo_root() -> Path:
    """Path to the checkout containing this file."""
    return Path(__file__).resolve().parent.parent


def repo_model_dir(size: str) -> Path:
    """Conventional in-repo location for vendored weights of ``size``."""
    return repo_root() / VENDOR_SUBDIR / f"faster-whisper-{size}"


def vendored_model_dir(size: str = DEFAULT_MODEL) -> Path | None:
    """Return usable vendored weights, preferring an explicit override.

    Checks ``WHISPER_MODEL_DIR`` first, then the in-repo ``models/`` directory.
    Committing weights to the repo works even when huggingface.co is blocked,
    because the repo itself is cloned over an allowed git transport.
    """
    raw = os.environ.get("WHISPER_MODEL_DIR")
    if raw:
        path = Path(raw)
        if is_model_dir(path):
            return path

    in_repo = repo_model_dir(size)
    return in_repo if is_model_dir(in_repo) else None


def ensure_model(size: str = DEFAULT_MODEL, *, allow_download: bool = True) -> Path:
    """Return a local path to the weights for ``size``.

    Raises :class:`ModelUnavailableError` with remediation steps when the
    weights are missing and cannot be fetched.
    """
    vendored = vendored_model_dir(size)
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
    size = repo.rsplit("-", 1)[-1]
    return (
        f"Whisper weights for {repo} are not available offline.\n"
        "Fix it in one of three ways:\n"
        "  1. Allow 'huggingface.co' in this environment's network policy, then start a "
        "NEW session so the hook can cache the weights (a running container keeps the "
        "policy it booted with).\n"
        f"  2. Commit the weights to this repo at {VENDOR_SUBDIR}/faster-whisper-{size}/ "
        "-- see the 'Vendoring weights' section of README.md.\n"
        f"  3. Point WHISPER_MODEL_DIR at a CTranslate2 model directory "
        f"(it must contain {', '.join(REQUIRED_FILES)})."
    )
