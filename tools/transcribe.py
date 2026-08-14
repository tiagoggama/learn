#!/usr/bin/env python3
"""Transcribe an audio file locally with faster-whisper.

    python3 tools/transcribe.py recording.m4a
    python3 tools/transcribe.py recording.m4a --model small --language pt --format srt

Runs entirely offline once the weights are cached; see tools/whisper_setup.py
for how weights are resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Running this file as a script puts tools/ on sys.path instead of the repo
# root, so make the package importable either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.whisper_setup import (  # noqa: E402
    DEFAULT_MODEL,
    MODEL_SIZES,
    ModelUnavailableError,
    ensure_model,
)

FORMATS = ("txt", "srt", "json")


@dataclass(frozen=True)
class Segment:
    """One timestamped chunk of speech."""

    start: float
    end: float
    text: str


def format_timestamp(seconds: float, *, millis_sep: str = ",") -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm`` (SRT style by default)."""
    if seconds < 0:
        raise ValueError("timestamp must be non-negative")
    total_ms = round(seconds * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def to_txt(segments: list[Segment]) -> str:
    """Render segments as ``[HH:MM:SS.mmm] text`` lines."""
    return "\n".join(
        f"[{format_timestamp(s.start, millis_sep='.')}] {s.text.strip()}" for s in segments
    )


def to_srt(segments: list[Segment]) -> str:
    """Render segments as an SRT subtitle document."""
    blocks = [
        f"{i}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{s.text.strip()}"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n\n".join(blocks)


def to_json(segments: list[Segment]) -> str:
    """Render segments as a JSON array of ``{start, end, text}`` objects."""
    payload = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render(segments: list[Segment], fmt: str) -> str:
    """Render ``segments`` in one of :data:`FORMATS`."""
    renderers = {"txt": to_txt, "srt": to_srt, "json": to_json}
    if fmt not in renderers:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return renderers[fmt](segments)


def transcribe(
    audio: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    compute_type: str = "int8",
) -> tuple[list[Segment], str]:
    """Transcribe ``audio``, returning its segments and the detected language."""
    from faster_whisper import WhisperModel

    model_path = ensure_model(model_size)
    model = WhisperModel(str(model_path), device="cpu", compute_type=compute_type)
    raw_segments, info = model.transcribe(str(audio), language=language, vad_filter=True)
    segments = [Segment(start=s.start, end=s.end, text=s.text) for s in raw_segments]
    return segments, info.language


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="path to the audio file")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_SIZES)
    parser.add_argument("--language", default=None, help="ISO code; omit to auto-detect")
    parser.add_argument("--format", default="txt", choices=FORMATS, dest="fmt")
    parser.add_argument("--output", type=Path, default=None, help="write here instead of stdout")
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2

    try:
        segments, language = transcribe(args.audio, model_size=args.model, language=args.language)
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    text = render(segments, args.fmt)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"detected language: {language}", file=sys.stderr)
        print(f"wrote {len(segments)} segments to {args.output}", file=sys.stderr)
    else:
        print(f"detected language: {language}", file=sys.stderr)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
