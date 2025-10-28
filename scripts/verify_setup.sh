#!/usr/bin/env bash
# Quick verification script for the Local AI workshop environment
set -e

echo "== Basic OS info =="
uname -a || true

echo "\n== Python & Jupyter =="
python --version || python3 --version || true
which jupyter || which jupyter-notebook || true

echo "\n== Mamba/Conda =="
which mamba || which conda || true

echo "\n== Docker =="
docker --version || true

echo "\n== NVIDIA GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found. If you have an NVIDIA GPU, ensure drivers are installed."
fi

echo "\n== End of checks =="
