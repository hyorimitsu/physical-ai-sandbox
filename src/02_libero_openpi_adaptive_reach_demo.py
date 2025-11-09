"""
Adaptive reach-and-grasp demo for LIBERO + OpenPI (e.g., pi0/pi05).

- Loads a LIBERO task (supports nested or flat layouts)
- Creates an OffScreenRenderEnv for headless simulation
- Runs an OpenPI policy with:
  - horizon-based MPC-style rollout
  - adaptive translation gain when motion stalls
  - pre-macro for initial approach
  - grasp assist finite-state machine
- Streams a frontview camera video directly to MP4
"""

import builtins
import importlib
import os
from pathlib import Path

import imageio
import numpy as np

# ----------------------------------------------------------------------
# 0. Safety: suppress any interactive prompts (e.g., LIBERO dataset path)
# ----------------------------------------------------------------------
builtins.input = lambda *_, **__: "N"

# ----------------------------------------------------------------------
# 1. Helper functions
# ----------------------------------------------------------------------
def get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def find_bddl_root():
    """Locate the 'bddl_files' directory under LIBERO or current working dir."""
    import libero
    roots = []
    if getattr(libero, "__path__", None):
        roots += [Path(p).resolve() for p in libero.__path__]
    roots.append(Path.cwd().resolve())
    for base in roots:
        for p in base.rglob("bddl_files"):
            if p.is_dir():
                return p.resolve()
    return None


def np_uint8_rgb(arr):
    """Normalize an array to uint8 RGB (H, W, 3)."""
    arr = np.asarray(arr)
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    # Normalize non-uint8
    if arr.dtype != np.uint8:
        scale = 255 if arr.max() <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    # Drop alpha channel
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def extract_state8(obs):
    """Extract an 8-element state vector from a dict-like observation."""
    if isinstance(obs, dict):
        for k in ["state", "robot_state", "qpos", "lowdim", "proprio", "agent_state"]:
            if k in obs:
                a = np.asarray(obs[k], dtype=np.float32).ravel()
                out = np.zeros(8, np.float32)
                n = min(8, a.size)
                out[:n] = a[:n]
                return out
    return np.zeros(8, np.float32)


def prompt_from_task(task):
    """Derive a text prompt from a LIBERO task object."""
    for attr in ("language", "language_instruction", "instruction", "goal"):
        if hasattr(task, attr):
            val = getattr(task, attr)
            if isinstance(val, str) and val:
                return val
    return Path(task.bddl_file).stem.replace("_", " ")


# ----------------------------------------------------------------------
# 2. LIBERO import (supports both nested/flat layouts)
# ----------------------------------------------------------------------
import libero

try:
    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")
except Exception:
    benchmark = importlib.import_module("libero.benchmark")
    envs = importlib.import_module("libero.envs")

bddl_root = find_bddl_root()
if bddl_root is None:
    raise SystemExit("bddl_files not found. Please ensure LIBERO datasets exist.")

get_bench_dict = getattr(benchmark, "get_benchmark_dict")
OffScreenRenderEnv = getattr(envs, "OffScreenRenderEnv")

suite = os.environ.get("LIBERO_SUITE", "libero_object")
task_id = get_env_int("LIBERO_TASK_ID", 0)
bench = get_bench_dict()[suite]()
task = bench.get_task(task_id)

bddl = bddl_root / task.problem_folder / task.bddl_file
if not bddl.exists():
    candidates = list(bddl_root.rglob(task.bddl_file))
    if not candidates:
        raise SystemExit(f"BDDL not found: {bddl}")
    bddl = candidates[0]

# ----------------------------------------------------------------------
# 3. Environment setup
# ----------------------------------------------------------------------
# Allow camera resolution to be overridden via environment variables
CAM_H = get_env_int("CAMERA_HEIGHT", 160)  # default: 160 (lighter than 224)
CAM_W = get_env_int("CAMERA_WIDTH", 160)   # default: 160

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl),
    camera_heights=CAM_H,
    camera_widths=CAM_W,
)

try:
    env.seed(0)
except Exception:
    pass

env.reset()
try:
    init_states = bench.get_task_init_states(task_id)
    env.set_init_state(init_states[0])
except Exception:
    pass


def render_pair():
    """Render a base + wrist RGB pair from the simulator (no flip)."""
    try:
        base = env.sim.render(
            camera_name="frontview", height=CAM_H, width=CAM_W, depth=False
        )
    except Exception:
        base = env.sim.render(
            camera_name="topview", height=CAM_H, width=CAM_W, depth=False
        )
    try:
        wrist = env.sim.render(
            camera_name="robot0_eye_in_hand", height=CAM_H, width=CAM_W, depth=False
        )
    except Exception:
        wrist = base
    return np_uint8_rgb(base), np_uint8_rgb(wrist)


# ----------------------------------------------------------------------
# 4. OpenPI policy setup
# ----------------------------------------------------------------------
from openpi.shared import download
from openpi.policies import policy_config
from openpi.training import config as cfg

# Use smaller pi0.5 Libero config by default to reduce memory usage
USE_PI = os.environ.get("OPENPI_CONF", "pi05_libero")

conf = cfg.get_config(USE_PI)
ckpt = download.maybe_download(f"gs://openpi-assets/checkpoints/{USE_PI}")
policy = policy_config.create_trained_policy(conf, ckpt)

prompt = prompt_from_task(task)

# ----------------------------------------------------------------------
# 5. Hyperparameters & controls
# ----------------------------------------------------------------------
out_path = os.environ.get("OUT_MP4", "02_libero_openpi_adaptive_reach_demo.mp4")

# Stream frames directly to video writer (avoid accumulating them in RAM)
video_writer = imageio.get_writer(out_path, fps=24, codec="libx264", quality=8)
frame_count = 0

TOTAL_STEPS = get_env_int("TOTAL_STEPS", 1800)
HZN = get_env_int("HORIZON", 15)

# Translation limits from action_space (fallback to constant)
try:
    hi = env.action_space.high[:7]
    TRANS_MAX = float(os.environ.get("TRANS_MAX", str(0.95 * float(np.max(hi[:3])))))
except Exception:
    TRANS_MAX = get_env_float("TRANS_MAX", 2.2)

ROT_LOCK = get_env_int("ROT_LOCK", 1)

# Translation smoothing and biases
TRANS_SMOOTH = get_env_float("TRANS_SMOOTH", 0.25)
Z_BIAS = get_env_float("Z_BIAS", -0.05)
Z_APPROACH = get_env_float("Z_APPROACH", -0.02)

# Deadband for minimal translation
STEP_FLOOR = get_env_float("STEP_FLOOR", 0.12)

# Adaptive gain for translation magnitude
ADAPT_ON = get_env_int("ADAPT_ON", 1)
ADAPT_GAIN0 = get_env_float("ADAPT_GAIN0", 1.0)
ADAPT_GAIN_MAX = get_env_float("ADAPT_GAIN_MAX", 3.0)
ADAPT_STEP = get_env_float("ADAPT_STEP", 0.2)
STUCK_N = get_env_int("STUCK_N", 35)
STUCK_EPS = get_env_float("STUCK_EPS", 0.05)

# Pre-macro: bring end-effector into the workspace
PRE_STEPS = get_env_int("PRE_STEPS", 40)
PRE_X = get_env_float("PRE_X", +0.45)
PRE_Z = get_env_float("PRE_Z", -0.20)

# Grasp assist FSM thresholds
GRASP_ASSIST = get_env_int("GRASP_ASSIST", 1)
OPEN_TH = get_env_float("OPEN_TH", -0.02)
CLOSE_TH = get_env_float("CLOSE_TH", 0.02)
PUSH_STEPS = get_env_int("PUSH_STEPS", 22)
HOLD_STEPS = get_env_int("HOLD_STEPS", 36)
LIFT_STEPS = get_env_int("LIFT_STEPS", 45)
Z_PUSH = get_env_float("Z_PUSH", -0.08)
Z_LIFT = get_env_float("Z_LIFT", +0.12)

APPROACH, CLOSE_PUSH, LIFT_HOLD = 0, 1, 2
fsm_state = APPROACH
push_cnt = 0
hold_cnt = 0

prev_a = np.zeros(7, np.float32)
grip_state = -1.0

# Pseudo end-effector position (integrated from actions)
xyz_est = np.zeros(3, np.float32)
xyz_hist = [xyz_est.copy()]
adapt_gain = ADAPT_GAIN0


def safe_reset():
    """Reset environment and restore initial state if available."""
    env.reset()
    try:
        init_states = bench.get_task_init_states(task_id)
        env.set_init_state(init_states[0])
    except Exception:
        pass
    obs, _, _, _ = env.step([0.0] * 7)
    return obs


def apply_step(a):
    """Execute one action step, render + record a frame, and update pseudo position."""
    global prev_a, xyz_est, grip_state, frame_count

    # Clip action to valid range if available
    try:
        lo = env.action_space.low[:7]
        hi = env.action_space.high[:7]
        a = np.clip(a, lo, hi)
    except Exception:
        a = np.clip(a, -1.0, 1.0)

    # Step environment; handle early termination
    try:
        obs, reward, done, info = env.step(a.tolist())
    except ValueError as e:
        if "terminated" in str(e).lower():
            obs = safe_reset()
            done = False
        else:
            raise

    # Render frontview frame for video
    try:
        frame = env.sim.render(
            camera_name="frontview", height=CAM_H, width=CAM_W, depth=False
        )
    except Exception:
        frame = None

    if frame is not None:
        # Drop alpha channel if present
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Flip vertically: MuJoCo images are upside down by default
        frame = np.flipud(frame)
        # Stream to video file
        video_writer.append_data(frame)
        frame_count += 1

    prev_a = a
    xyz_est[:] = xyz_est + a[:3] * 0.5
    xyz_hist.append(xyz_est.copy())
    return obs, done


# ----------------------------------------------------------------------
# 6. Initial observation and pre-macro
# ----------------------------------------------------------------------
obs, _, done, _ = env.step([0.0] * 7)

# Move forward & down a bit to get closer to the workspace
for _ in range(PRE_STEPS):
    a = np.zeros(7, np.float32)
    a[0] = PRE_X
    a[2] = PRE_Z
    if ROT_LOCK:
        a[3:6] = 0.0
    obs, done = apply_step(a)
    if done:
        break


# ----------------------------------------------------------------------
# 7. Main rollout loop
# ----------------------------------------------------------------------
for t in range(0, TOTAL_STEPS, HZN):
    if done:
        obs = safe_reset()
        done = False

    base_img, wrist_img = render_pair()
    state8 = extract_state8(obs)

    sample = {
        "observation/state": state8,
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "prompt": prompt,
    }

    plan = policy.infer(sample)["actions"][:HZN]  # (HZN, 7)

    # Adaptive gain update based on recent motion
    if ADAPT_ON and len(xyz_hist) > STUCK_N:
        progress = np.linalg.norm(xyz_est - xyz_hist[-STUCK_N])
        if progress < STUCK_EPS and adapt_gain < ADAPT_GAIN_MAX:
            adapt_gain = min(ADAPT_GAIN_MAX, adapt_gain + ADAPT_STEP)

    for a in plan:
        a = np.asarray(a, dtype=np.float32)

        # --- Translation: scale, deadband, smoothing, biases ---
        a[:3] *= adapt_gain

        # Apply deadband per axis
        for j in range(3):
            if abs(a[j]) < STEP_FLOOR:
                a[j] = np.sign(a[j]) * STEP_FLOOR

        a[:3] = np.clip(a[:3], -TRANS_MAX, TRANS_MAX)

        # Constant vertical bias
        a[2] += Z_BIAS

        # Exponential smoothing with previous action
        a[:3] = TRANS_SMOOTH * prev_a[:3] + (1.0 - TRANS_SMOOTH) * a[:3]

        # Additional approach bias along Z
        a[2] += Z_APPROACH

        # Lock rotation if requested
        if ROT_LOCK:
            a[3:6] = 0.0

        # --- Gripper grasp assist FSM ---
        g_raw = float(a[6])

        if GRASP_ASSIST:
            if fsm_state == APPROACH:
                if g_raw > CLOSE_TH:
                    fsm_state = CLOSE_PUSH
                    push_cnt = 0
                    grip_state = 1.0
                elif g_raw < OPEN_TH:
                    grip_state = -1.0

            elif fsm_state == CLOSE_PUSH:
                grip_state = 1.0
                a[2] += Z_PUSH
                push_cnt += 1
                if push_cnt >= PUSH_STEPS:
                    fsm_state = LIFT_HOLD
                    hold_cnt = 0

            elif fsm_state == LIFT_HOLD:
                grip_state = 1.0
                a[2] += Z_LIFT
                hold_cnt += 1
                if hold_cnt >= LIFT_STEPS:
                    fsm_state = APPROACH
        else:
            # Simple threshold-based open/close
            if g_raw > CLOSE_TH:
                grip_state = 1.0
            elif g_raw < OPEN_TH:
                grip_state = -1.0

        a[6] = grip_state

        obs, done = apply_step(a)
        if done:
            break


# ----------------------------------------------------------------------
# 8. Cleanup and video export
# ----------------------------------------------------------------------
env.close()
video_writer.close()

if frame_count > 0:
    print("[OK] Saved:", out_path, "frames:", frame_count)
else:
    print("[NG] No frames captured. Check rendering / MUJOCO_GL / camera names.")
