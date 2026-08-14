# learn

Local, offline audio transcription for Claude Code on the web, wired up through a
`SessionStart` hook.

## Why this exists

Claude Code sessions have no native audio input, and this repo's remote environment
ships without `ffmpeg` or any speech-to-text tooling. This adds both:

- a `SessionStart` hook that installs [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  and warms the model cache, so the work happens once and is baked into the cached
  container image rather than repeated every session;
- `tools/transcribe.py`, a small CLI that turns an audio file into text, SRT, or JSON.

`faster-whisper` decodes audio through PyAV, so no system `ffmpeg` binary is needed.

## Usage

```bash
python3 tools/transcribe.py recording.m4a                      # timestamped text
python3 tools/transcribe.py recording.m4a --language pt        # skip auto-detect
python3 tools/transcribe.py recording.m4a --format srt --output transcripts/rec.srt
python3 tools/transcribe.py recording.m4a --model medium       # tiny|base|small|medium|large-v3
```

Language is auto-detected when `--language` is omitted, and the detected code is
reported on stderr.

## Network requirement (important)

Two different hosts are involved, and only one of them is reliably reachable:

| What | Host | Status in this environment |
| --- | --- | --- |
| Python packages | `pypi.org`, `files.pythonhosted.org` | Reachable — they bypass the egress proxy |
| Whisper weights | `huggingface.co` | **Blocked** — the proxy answers `403` to `CONNECT` |

So the hook installs the code successfully, but cannot fetch the weights. It treats
that as a warning rather than an error, and the session still starts. Transcription
then fails with an explicit message until you do one of the following:

1. **Allow `huggingface.co`** in the environment's network policy
   ([docs](https://code.claude.com/docs/en/claude-code/claude-code-on-the-web)), then
   start a fresh session. The hook downloads the weights once and the cached image
   keeps them, so later sessions work even if the host is blocked again.
2. **Vendor the weights** into the repo — see below. This works with
   `huggingface.co` permanently blocked, because the repo arrives over git.

Note that a *running* container keeps the network policy it booted with, so
allowing the host does nothing for a session that is already open. Start a new one.

## Vendoring weights

The weights are fetched once on your own machine and committed, after which every
session gets them from the clone. The hook detects
`models/faster-whisper-<size>/` automatically and exports `WHISPER_MODEL_DIR`.

`model.bin` is larger than GitHub's 100 MB per-file limit for every size except
`tiny`, so `.gitattributes` tracks it with Git LFS.

```bash
# On your machine, in a clone of this repo:
pip install huggingface_hub
git lfs install

python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-small", local_dir="models/faster-whisper-small")
PY

git add .gitattributes models/faster-whisper-small
git commit -m "Vendor faster-whisper-small weights"
git push
```

Pick the size to match the clone cost you are willing to pay every session:

| Size | `model.bin` | Needs LFS |
| --- | --- | --- |
| `tiny` | ~75 MB | no |
| `base` | ~145 MB | yes |
| `small` | ~480 MB | yes |

Then set `WHISPER_MODEL` to that size (the hook defaults to `small`).

If a session ever reports weights that will not load, the usual cause is a clone
without `git lfs pull`, which leaves pointer stubs in place of the real files.
`tools/whisper_setup.py` detects that case and reports it rather than failing
inside CTranslate2.

Check the current policy from inside a session with:

```bash
curl -sS "$HTTPS_PROXY/__agentproxy/status"
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL` | `small` | Model size the hook pre-caches |
| `WHISPER_CACHE` | `/opt/whisper-cache` | Weight cache; outside the repo, which is re-cloned each session |
| `WHISPER_MODEL_DIR` | unset | Explicit path to vendored weights; wins over the cache |

`small` is the default because it fits the 4-CPU remote container: roughly 2–4 minutes
of wall time per 5 minutes of audio at `int8`. Use `medium` or `large-v3` for accuracy
if you are willing to wait considerably longer.

## Development

```bash
python3 -m pip install -r requirements.txt
python3 -m ruff check .
python3 -m pytest -q
```

The tests cover timestamp/format rendering and weight resolution; they do not download
models or run inference, so they pass offline.
