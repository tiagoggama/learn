#!/bin/bash
# SessionStart hook: make local audio transcription work without network access.
#
# Installs faster-whisper from PyPI and warms the Whisper weight cache. The
# container image is snapshotted after this hook finishes, so both survive into
# later sessions. The cache deliberately lives outside the repo, which is
# re-cloned fresh every session.
set -euo pipefail

# Only relevant for Claude Code on the web; local checkouts manage their own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WHISPER_CACHE="${WHISPER_CACHE:-/opt/whisper-cache}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"

mkdir -p "$WHISPER_CACHE"
export HF_HOME="$WHISPER_CACHE"
export WHISPER_CACHE WHISPER_MODEL

# pypi.org and files.pythonhosted.org bypass the egress proxy, so this works
# under every network policy. Plain `install` (not a locked sync) so repeated
# runs are no-ops against the cached image.
python3 -m pip install --quiet --disable-pip-version-check --root-user-action=ignore \
  -r "$PROJECT_DIR/requirements.txt"

# Persist for the session so `python3 tools/transcribe.py` finds the cache.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export HF_HOME=\"$WHISPER_CACHE\""
    echo "export WHISPER_CACHE=\"$WHISPER_CACHE\""
    echo "export WHISPER_MODEL=\"$WHISPER_MODEL\""
  } >> "$CLAUDE_ENV_FILE"
fi

# Best-effort: weights come from huggingface.co, which some network policies
# block. A miss must not fail the session -- transcribe.py reports it later
# with remediation steps.
if PYTHONPATH="$PROJECT_DIR" python3 -c "
from tools.whisper_setup import ensure_model
print(ensure_model('${WHISPER_MODEL}'))
" 2>/dev/null; then
  echo "whisper: '${WHISPER_MODEL}' weights cached in ${WHISPER_CACHE}"
else
  echo "whisper: faster-whisper installed, but '${WHISPER_MODEL}' weights could not be" \
       "fetched (huggingface.co unreachable under this environment's network policy)." \
       "Transcription stays unavailable until huggingface.co is allowed or" \
       "WHISPER_MODEL_DIR points at vendored weights. See README.md."
fi
