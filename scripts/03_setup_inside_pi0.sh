#!/usr/bin/env bash

set -euxo pipefail

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "pi0" ]; then
  echo "❌ This script must be run inside the 'pi0' conda environment."
  echo "   Please run: conda activate pi0"
  exit 1
fi

echo "✅ Detected conda env: $CONDA_DEFAULT_ENV"
echo "   Using python: $(which python)"

# === 1. System packages ===

sudo apt update
sudo apt install -y \
  git tmux htop unzip curl ffmpeg \
  libegl1 libgles2 libgl1 libopengl0 libglib2.0-0

# === 2. Pin numpy to a version compatible with openpi ( < 2.0.0 ) ===

python -m pip install --upgrade "numpy<2.0.0,>=1.22.4"

# === 3. Install PyTorch (CUDA 12.1) ===

python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# === 4. Base dependencies (OpenPI + LIBERO common) ===

python -m pip install \
  mujoco mujoco-python-viewer \
  gymnasium gym-aloha \
  "imageio[ffmpeg]" websockets \
  jupyterlab pytest uv

python -m pip install -U \
  "gym==0.26.2" gym-notices \
  "robosuite==1.4.0" \
  bddl "easydict>=1.10" termcolor yacs \
  matplotlib Pillow opencv-python-headless imageio

# === 5. Clone and set up OpenPI (editable install in pi0) ===

cd "$HOME"
if [ ! -d openpi ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
fi

cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

python -m ipykernel install --user --name pi0 --display-name "Python (pi0)"

# === 6. Clone and set up LIBERO (editable install in pi0) ===

cd "$HOME"
if [ ! -d LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi

cd LIBERO
python -m pip install -e .

# === 7. Add LIBERO to PYTHONPATH for all future shells ===

if ! grep -q "LIBERO_ROOT=" "$HOME/.bashrc" 2>/dev/null; then
  {
    echo 'export LIBERO_ROOT="$HOME/LIBERO"'
    echo 'export PYTHONPATH="$LIBERO_ROOT:$PYTHONPATH"'
  } >> "$HOME/.bashrc"
fi

# === 8. Environment variables for headless rendering & XLA ===

if ! grep -q "MUJOCO_GL=egl" "$HOME/.bashrc" 2>/dev/null; then
  echo 'export MUJOCO_GL=egl' >> "$HOME/.bashrc"
fi
if ! grep -q "unset DISPLAY" "$HOME/.bashrc" 2>/dev/null; then
  echo 'unset DISPLAY' >> "$HOME/.bashrc"
fi

{
  echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false'
  echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6'
  echo 'export MPLBACKEND=Agg'
} >> "$HOME/.bashrc"

echo
echo "✅ pi0 environment is ready."
echo "Next steps:"
echo "  1) open a new shell or run: source ~/.bashrc"
echo "  2) run: conda activate pi0"
echo "  3) run your Python script, e.g.:"
echo "       python your_script.py"
echo "     or:"
echo "       python - <<'PY'"
echo "       # your long LIBERO + OpenPI script here"
echo "       PY"
