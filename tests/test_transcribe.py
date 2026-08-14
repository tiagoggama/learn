from __future__ import annotations

import json

import pytest

from tools.transcribe import Segment, format_timestamp, render, to_json, to_srt, to_txt
from tools.whisper_setup import (
    LFS_POINTER_MAGIC,
    is_lfs_pointer,
    is_model_dir,
    model_repo,
    repo_model_dir,
    vendored_model_dir,
)


def _write_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(b"\x00" * 2048)
    (path / "config.json").write_text("{}")
    return path

SEGMENTS = [
    Segment(start=0.0, end=2.5, text=" Bom dia a todos. "),
    Segment(start=2.5, end=3661.75, text="Vamos comecar."),
]


def test_format_timestamp_pads_all_fields():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.75) == "01:01:01,750"


def test_format_timestamp_honours_separator():
    assert format_timestamp(2.5, millis_sep=".") == "00:00:02.500"


def test_format_timestamp_rejects_negative():
    with pytest.raises(ValueError):
        format_timestamp(-1)


def test_to_txt_strips_and_timestamps_each_line():
    assert to_txt(SEGMENTS).splitlines() == [
        "[00:00:00.000] Bom dia a todos.",
        "[00:00:02.500] Vamos comecar.",
    ]


def test_to_srt_numbers_blocks_from_one():
    assert to_srt(SEGMENTS).split("\n\n")[0] == (
        "1\n00:00:00,000 --> 00:00:02,500\nBom dia a todos."
    )


def test_to_json_round_trips():
    payload = json.loads(to_json(SEGMENTS))
    assert payload[0] == {"start": 0.0, "end": 2.5, "text": "Bom dia a todos."}


def test_render_rejects_unknown_format():
    with pytest.raises(ValueError):
        render(SEGMENTS, "vtt")


def test_model_repo_maps_size_to_ct2_repo():
    assert model_repo("small") == "Systran/faster-whisper-small"


def test_model_repo_rejects_unknown_size():
    with pytest.raises(ValueError):
        model_repo("enormous")


def test_is_model_dir_requires_both_files(tmp_path):
    assert not is_model_dir(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"\x00" * 2048)
    assert not is_model_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert is_model_dir(tmp_path)


def test_is_lfs_pointer_detects_unfetched_weights(tmp_path):
    pointer = tmp_path / "model.bin"
    pointer.write_bytes(LFS_POINTER_MAGIC + b"\noid sha256:abc\nsize 123\n")
    assert is_lfs_pointer(pointer)

    real = tmp_path / "real.bin"
    real.write_bytes(b"\x00" * 2048)
    assert not is_lfs_pointer(real)


def test_is_model_dir_rejects_lfs_pointer(tmp_path):
    _write_model(tmp_path)
    (tmp_path / "model.bin").write_bytes(LFS_POINTER_MAGIC + b"\noid sha256:abc\n")
    assert not is_model_dir(tmp_path)


def test_vendored_model_dir_ignores_incomplete_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path))
    assert vendored_model_dir() is None

    _write_model(tmp_path)
    assert vendored_model_dir() == tmp_path


def test_vendored_model_dir_unset(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL_DIR", raising=False)
    assert vendored_model_dir() is None


def test_repo_model_dir_uses_models_subdir():
    assert repo_model_dir("small").parts[-2:] == ("models", "faster-whisper-small")


def test_vendored_model_dir_falls_back_to_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL_DIR", str(tmp_path / "does-not-exist"))
    in_repo = _write_model(tmp_path / "models" / "faster-whisper-small")
    monkeypatch.setattr("tools.whisper_setup.repo_root", lambda: tmp_path)
    assert vendored_model_dir("small") == in_repo
