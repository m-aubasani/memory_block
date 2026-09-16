import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [PROJECT_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .model import AlignedInjectedLLM, GatedCrossAttention, ConstraintEncoder
from .inference import InjectedGenerator
from .train import train_model

__all__ = [
    "AlignedInjectedLLM",
    "GatedCrossAttention",
    "ConstraintEncoder",
    "InjectedGenerator",
    "train_model",
]
