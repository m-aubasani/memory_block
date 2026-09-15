import math
import torch
import torch.nn as nn
from typing import Union, Tuple


class SteeringHook:
    """
    Wraps a fixed steering vector and applies it to a layer's output via a forward hook.

    Modes:
      - mode="add":
          h_new = h + coefficient * vector

      - mode="rotate":
          norm-preserving spherical interpolation:
            orig_norm = ||h||_2 (per-token, last dim)
            h_hat = h / orig_norm
            v_hat = vector / ||vector||_2
            theta = angle_rad  (0 = no steering, pi/2 = fully toward vector direction)
            h_rot = h_hat * cos(theta) + v_hat * sin(theta)
            h_new = h_rot * orig_norm
    """

    def __init__(
        self,
        vector: torch.Tensor,
        mode: str = "add",
        coefficient: float = 1.0,
        angle_rad: float = 0.0,
    ):
        if mode not in ("add", "rotate"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'add' or 'rotate'.")

        self.vector = vector.detach().squeeze()
        self.mode = mode
        self.coefficient = float(coefficient)
        self.angle_rad = float(angle_rad)

    def __call__(
        self,
        module: nn.Module,
        args: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        orig_dtype = hidden_states.dtype
        device = hidden_states.device

        # Perform vector math in float32 for numerical stability
        h_f32 = hidden_states.to(dtype=torch.float32)
        v_f32 = self.vector.to(device=device, dtype=torch.float32)

        if self.mode == "add":
            # Broadcast vector across batch and sequence dimensions
            h_new_f32 = h_f32 + (self.coefficient * v_f32)
        elif self.mode == "rotate":
            # Norm-preserving spherical rotation per token along the hidden dimension
            orig_norm = torch.norm(h_f32, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            h_hat = h_f32 / orig_norm

            v_norm = torch.norm(v_f32, p=2).clamp_min(1e-8)
            v_hat = v_f32 / v_norm

            cos_theta = math.cos(self.angle_rad)
            sin_theta = math.sin(self.angle_rad)

            h_rot = h_hat * cos_theta + v_hat * sin_theta
            h_new_f32 = h_rot * orig_norm
        else:
            h_new_f32 = h_f32

        h_new = h_new_f32.to(dtype=orig_dtype)

        if is_tuple:
            return (h_new,) + output[1:]
        return h_new


def register_steering_hook(base_model, layer_idx: int, hook: SteeringHook):
    """
    Attaches hook to base_model's transformer layer at layer_idx and returns
    the hook handle for removal.
    """
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        target_layer = base_model.model.layers[layer_idx]
    elif hasattr(base_model, "base_model") and hasattr(base_model.base_model, "model") and hasattr(base_model.base_model.model, "layers"):
        target_layer = base_model.base_model.model.layers[layer_idx]
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        target_layer = base_model.transformer.h[layer_idx]
    else:
        raise AttributeError(f"Could not resolve transformer layers from model of type {type(base_model)}.")

    return target_layer.register_forward_hook(hook)
