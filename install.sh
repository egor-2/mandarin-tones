#!/bin/bash
# Install script for Chinese Tone Guessing Game
# Requires: macOS on Apple Silicon (M1/M2/M3/M4)
#
# Packages installed:
#   mlx-audio==0.4.3       - MLX-based text-to-speech
#   soundfile==0.13.1      - Audio file I/O
#   numpy==2.4.5           - Numerical computing
#   misaki[zh]==0.9.4      - Chinese text processing
#   pypinyin               - Chinese character to pinyin conversion

set -e

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install \
    mlx-audio==0.4.3 \
    soundfile==0.13.1 \
    numpy==2.4.5 \
    "misaki[zh]==0.9.4" \
    pypinyin

echo "Downloading TTS model..."
python -c "
from mlx_audio.tts.utils import load_model
load_model('mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16')
print('Model downloaded OK')
"

echo ""
echo "Installation complete!"
echo "Run the game with: python start.py"
