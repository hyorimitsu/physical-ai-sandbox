from openpi.shared import download
from openpi.policies import policy_config
from openpi.training import config as cfg
import numpy as np

# Load model configuration and checkpoint
conf = cfg.get_config("pi0_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")

# Create a trained policy instance
policy = policy_config.create_trained_policy(conf, ckpt)

# Dummy observation input
# (in practice, this would contain camera images, wrist images, and state vectors, etc.)
sample = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros((8,), dtype=np.float32),
    "prompt": "pick and place the cube",
}

# Inference (generate actions)
out = policy.infer(sample)
print("actions:", out["actions"].shape)
