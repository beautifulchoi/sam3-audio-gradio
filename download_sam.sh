#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$ROOT_DIR" >/dev/null

# 1) sam-audio
if [ ! -d sam-audio ]; then
  git clone https://github.com/facebookresearch/sam-audio.git
fi
pushd sam-audio >/dev/null
python -m pip install -e .
popd >/dev/null

# 2) sam3
if [ ! -d sam3 ]; then
  git clone https://github.com/facebookresearch/sam3.git
fi
pushd sam3 >/dev/null
python -m pip install -e .
popd >/dev/null

popd >/dev/null
echo "✔ sam-audio and sam3 installed."