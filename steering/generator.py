from contextlib import contextmanager
from typing import Dict, Optional, List
import torch
from .steering_hook import SteeringHook, register_steering_hook


class SteeredGenerator:
    """
    Generation wrapper for steered inference using activation addition / rotation hooks.
    Mirrors InjectedGenerator's interface.
    """

    def __init__(self, base_model, hooks: Dict[int, SteeringHook]):
        self.base_model = base_model
        self.hooks = hooks
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach_hooks(self):
        """
        Attaches all configured steering hooks to their respective layers.
        """
        self.remove_hooks()
        for layer_idx, hook in self.hooks.items():
            handle = register_steering_hook(self.base_model, layer_idx, hook)
            self.handles.append(handle)

    def remove_hooks(self):
        """
        Removes all active steering hook handles.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @contextmanager
    def steering_context(self):
        """
        Context manager to activate steering hooks during a generation block and clean up afterwards.
        """
        try:
            self.attach_hooks()
            yield
        finally:
            self.remove_hooks()

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Generates text using the base model with steering hooks attached.
        """
        with self.steering_context():
            if hasattr(self.base_model, "base_model"):
                target = self.base_model.base_model
            else:
                target = self.base_model
            return target.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
