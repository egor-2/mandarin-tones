#!/bin/bash
# Install script for Chinese Tone Guessing Game
# Requires: macOS on Apple Silicon (M1/M2/M3/M4), Python 3.11

set -e

PYTHON="/opt/homebrew/bin/python3.11"
VENV="ml_env"

if [ ! -f "$PYTHON" ]; then
    echo "Python 3.11 not found at $PYTHON"
    echo "Install with: brew install python@3.11"
    exit 1
fi

echo "Creating virtual environment '$VENV'..."
$PYTHON -m venv ~/$VENV

echo "Upgrading pip..."
~/$VENV/bin/pip install --upgrade pip

echo "Installing dependencies..."
~/$VENV/bin/pip install \
    mlx-audio==0.4.3 \
    soundfile==0.13.1 \
    numpy==2.4.5 \
    "misaki[zh]==0.9.4" \
    pypinyin

echo "Downloading TTS model..."
~/$VENV/bin/python -c "
from mlx_audio.tts.utils import load_model
load_model('mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16')
print('Model downloaded OK')
"

echo ""
echo "Installation complete!"
echo "Run the game with: ~/$VENV/bin/python start.py"
