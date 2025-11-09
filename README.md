Physical AI Sandbox — OpenPI + LIBERO Integration
---

A **Physical AI sandbox** that integrates [OpenPI](https://github.com/Physical-Intelligence/openpi) (by *Physical Intelligence*) with the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation environment for **adaptive reach-and-grasp demonstrations**.

This project provides a full reproducible environment for running **Vision-Language-Action (VLA)** models in simulation on GPU-enabled cloud infrastructure.

**NOTE:** This setup is intended for **research and sandbox experiments**. For production or shared environments:

* Assign **minimal IAM scopes** to the service account
* Restrict inbound access via **VPC firewall rules**
* Use **preemptible instances** for cost-efficient workloads


## Overview

This sandbox demonstrates advanced robotic manipulation using:

- **OpenPI** — Vision-Language-Action models (`π₀`, `π₀.₅`) for embodied reasoning and control  
- **LIBERO** — Lifelong Robot Learning benchmark suite for diverse manipulation tasks  
- **Adaptive Control** — MPC-style rollout with adaptive translation gains and grasp-assist FSM  
- **Cloud Infrastructure** — GPU-enabled Google Cloud environment provisioned via Terraform


## Features

- **Multi-modal Policy Integration** — Vision-Language-Action models for robotic manipulation  
- **Adaptive Reach-and-Grasp** — Horizon-based MPC with automatic gain adjustment when stalled  
- **Grasp-Assist FSM** — Finite-state machine for approach / push / lift cycles  
- **Headless Video Recording** — Streams frontview camera output directly to MP4 (no GUI required)  
- **Cloud-Ready** — Terraform configuration for Google Cloud GPU deployment  
- **Automated Setup** — Scripts for NVIDIA driver, Conda, PyTorch, and dependencies  


## Project Structure

```
physical-ai-sandbox/
├── src/
│   ├── 01_openpi_libero_smoke_test.py           # Basic OpenPI smoke test
│   └── 02_libero_openpi_adaptive_reach_demo.py  # Adaptive reach-and-grasp demo
├── scripts/
│   ├── 01_install_nvidia.sh                     # NVIDIA driver installation
│   ├── 02_create_conda_env.sh                   # Conda environment setup
│   └── 03_setup_inside_pi0.sh                   # Install OpenPI / LIBERO / deps
└── terraform/
    └── main.tf                                  # Google Cloud GPU VM configuration
```


## System Requirements

| Requirement | Recommended |
|--------------|-------------|
| OS | Ubuntu 22.04 / 24.04 LTS |
| GPU | NVIDIA L4 / A10 / A100 |
| CUDA | 12.1 (installed via PyTorch) |
| RAM | ≥ 16 GB (32 GB recommended) |
| Disk | ≥ 100 GB free |
| Python | 3.11 (via Conda) |


## Quick Start (Local or Google Cloud VM)

### 1. Run Setup Scripts

```bash
# 1. Install NVIDIA driver
./scripts/01_install_nvidia.sh
sudo reboot  # Required after installation

# 2. Create Conda environment
./scripts/02_create_conda_env.sh
source ~/.bashrc
conda activate pi0

# 3. Install dependencies (OpenPI, LIBERO, PyTorch, etc.)
./scripts/03_setup_inside_pi0.sh
source ~/.bashrc
```

### 2. Verify Installation

```bash
conda activate pi0
python src/01_openpi_libero_smoke_test.py
```

Expected output:

```
actions: (50, 7)
```

If you see this, the OpenPI model loaded successfully and produced a 50-step action sequence with 7-dimensional actions.

### 3. Run Adaptive Reach Demo

```bash
conda activate pi0
python src/02_libero_openpi_adaptive_reach_demo.py
```

This will generate:

```
02_libero_openpi_adaptive_reach_demo.mp4
```

containing the simulated robot performing **adaptive reach-and-grasp** with visual feedback.


## Configuration

The demo supports detailed tuning through environment variables.

### Task Configuration

```bash
export LIBERO_SUITE="libero_object"      # Benchmark suite
export LIBERO_TASK_ID=0                  # Task index
export OPENPI_CONF="pi05_libero"         # OpenPI model (pi0_libero / pi05_libero)
```

### Control Parameters

```bash
export TOTAL_STEPS=1800                  # Total rollout steps
export HORIZON=15                        # Inference horizon per iteration
export TRANS_MAX=2.2                     # Max translation magnitude (default: 0.95 * env.action_space)
export ROT_LOCK=1                        # Lock rotation (0/1)
```

### Adaptive Behavior

```bash
export ADAPT_ON=1                        # Enable adaptive translation gain (0/1)
export ADAPT_GAIN0=1.0                   # Initial translation gain
export ADAPT_GAIN_MAX=3.0                # Maximum adaptive gain
export STUCK_N=35                        # Step window for stall detection
export STUCK_EPS=0.05                    # Distance threshold for stall
```

### Grasp-Assist FSM

```bash
export GRASP_ASSIST=1                    # Enable grasp assist (0/1)
export PUSH_STEPS=22                     # Push phase length
export HOLD_STEPS=36                     # Hold phase length
export LIFT_STEPS=45                     # Lift phase length
```


## Cloud Deployment (Google Cloud)

Deploy a GPU-enabled VM via Terraform.

### Prerequisites

* Google Cloud account with billing enabled
* Terraform installed locally
* `gcloud` CLI configured with active credentials

### Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# (Optional) Preview configuration
terraform plan -var="project_id=your-google-cloud-project-id"

# Apply configuration
terraform apply -var="project_id=your-google-cloud-project-id"
```

### Default Configuration

| Setting      | Default                               |
| ------------ | ------------------------------------- |
| Machine type | `g2-standard-4` (4 vCPUs / 16 GB RAM) |
| GPU          | 1× NVIDIA L4                          |
| Region       | `asia-northeast1`                     |
| OS           | Ubuntu 24.04 LTS                      |
| Disk         | 100GB balanced persistent disk       |

### Connect and Initialize

```bash
# SSH into the deployed instance
gcloud compute ssh physical_ai --zone=asia-northeast1-a

# Clone repository and begin setup
git clone git@github.com:hyorimitsu/physical-ai-sandbox.git
cd physical-ai-sandbox
./scripts/01_install_nvidia.sh
# ...continue with setup steps
```


## Technical Details

### OpenPI Integration

* Supports **π₀** and **π₀.₅** pretrained models
* Accepts multi-modal inputs: RGB images, wrist-camera images, proprioceptive state vectors, and natural-language prompts
* Performs horizon-based inference and MPC-style rollouts

### LIBERO Environment

* Headless rendering via **MuJoCo EGL backend** (`MUJOCO_GL=egl`)
* Multiple camera views (`frontview`, `topview`, `eye-in-hand`)
* Configurable camera resolution and initial task states

### Adaptive Control

* **Translation Smoothing:** Exponential smoothing over previous actions
* **Deadband Control:** Ensures minimum translation per axis
* **Adaptive Gains:** Automatically increases gain when motion stalls
* **Vertical Biasing:** Adds approach and push/lift adjustments

### Grasp-Assist FSM

1. **APPROACH** — Wait for gripper trigger to start grasp
2. **CLOSE_PUSH** — Close gripper and push down
3. **LIFT_HOLD** — Lift and maintain grasp before releasing


## Troubleshooting

| Issue                           | Likely Cause             | Solution                                                |
| ------------------------------- | ------------------------ | ------------------------------------------------------- |
| **Process terminated (Killed)** | Out-of-memory on GPU     | Use smaller model (`OPENPI_CONF=pi05_libero`)           |
| **`bddl_files not found`**      | LIBERO dataset missing   | Ensure LIBERO repo includes `bddl_files/`               |
| **No frames in MP4**            | EGL rendering not active | Verify `MUJOCO_GL=egl` and `unset DISPLAY` in `.bashrc` |
