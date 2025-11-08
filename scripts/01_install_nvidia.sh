#!/usr/bin/env bash

set -euxo pipefail

sudo apt update
sudo apt install -y nvidia-driver-535

echo "✅ NVIDIA driver installation completed."
echo "👉 Please reboot your system: sudo reboot"
echo "👉 After reboot, verify with 'nvidia-smi' and then run 02_create_conda_env.sh."
