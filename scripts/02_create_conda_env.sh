#!/usr/bin/env bash

set -euxo pipefail

CONDA_ROOT="$HOME/miniconda"

# === 1. Install Miniconda if missing ===

if [ ! -d "$CONDA_ROOT" ]; then
  cd /tmp
  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_ROOT"
  echo 'export PATH=$HOME/miniconda/bin:$PATH' >> "$HOME/.bashrc"
fi

# === 2. Enable conda in this script ===

source "$CONDA_ROOT/etc/profile.d/conda.sh" || true

# === 3. Initialize conda for future shells (idempotent) ===

if ! grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null; then
  conda init bash || true
fi

# === 4. Accept Anaconda Terms of Service (required on recent conda) ===

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# === 5. Create env 'pi0' with Python 3.11 ===

if ! conda env list | grep -qE '^pi0\s'; then
  conda create -n pi0 python=3.11 -y
fi

echo
echo "✅ Conda environment 'pi0' has been created."
echo "Next steps:"
echo "  1) open a new shell or run: source ~/.bashrc"
echo "  2) run: conda activate pi0"
echo "  3) then run: ./03_setup_inside_pi0.sh"
