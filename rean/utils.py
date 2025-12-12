import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def rot_img(x, theta):
    """Rotate 2D images.
    Args:
        x : input images with shape [N, C, H, W]
        theta: angle (in radians) of rotation
    Returns:
        rotated images
    """
    device = x.device
    dtype = x.dtype

    # Rotation matrix (2 x 3) on the same device/dtype as x
    rot_mat = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta),  np.cos(theta), 0.0]],
        dtype=dtype,
        device=device
    )

    # Expand to N x 2 x 3
    rot_mat = rot_mat.unsqueeze(0).repeat(x.shape[0], 1, 1)

    # Grid on same device, with explicit align_corners
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False)

    # Use same align_corners here too
    x_rot = F.grid_sample(x, grid, align_corners=False)

    return x_rot

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_model(model_name):
    """Select and return the model class based on the model_name string."""
    if model_name.lower() == "plaincnn":
        from models.CNN import PlainCNN
        return PlainCNN
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def make_run_dir(model_name,
                 noise_type,
                 noise_params,
                 learning_rate,

                 ):
    gamma = noise_params.get("gamma")
    std = noise_params.get("std")
    run_dir = Path(f"runs/{model_name}_{noise_type}_std{std}_gamma{gamma}_lr{learning_rate}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
def load_run_data(run_dir):
    import json
    run_data_path = Path(run_dir) / "run_data.json"
    if not run_data_path.exists():
        raise FileNotFoundError(f"No run_data.json found in {run_dir}")
    with open(run_data_path, "r") as f:
        run_data = json.load(f)
    test_data_path = Path(run_dir) / "test_data.json"
    if test_data_path.exists():
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
    return run_data, test_data



def to_serializable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(type(obj))