# memory_block — Inference-Time Safety Alignment

Frozen-weight safety alignment for `Qwen/Qwen2.5-1.5B-Instruct`, no base-model fine-tuning.

Goal: raise adversarial safety rate (refuse / safe-complete jailbreaks) while keeping benign over-refusal low.

## Approaches

1. **`gated-cross-attention/` (trained):** learns a `GatedCrossAttention` + `ConstraintEncoder` module that injects `constitution.txt` axioms into mid-layer hidden states. Train on `PKU-SafeRLHF` contrastive pairs with loss masked to safe responses.
2. **`steering/` (training-free, current):** Contrastive Activation Addition (CAA) / difference-in-means steering vector (`mean(safe_acts) - mean(unsafe_acts)`) applied via forward hook at inference. Modes: `add` (`h + coef*v`) and `rotate` (norm-preserving). Sweep layer / coefficient / angle for best safety vs over-refusal tradeoff.

## Eval

- Unified checker: `allenai/wildguard` (`steering/wildguard_eval.py`).
- Adversarial Samples: JailbreakBench (default), HarmBench, PKU-SafeRLHF test-unsafe. 
- Benign Samples: XSTest (default), JBB-benign, PKU-safe. 
- Compares Baseline vs System-Prompt vs Steered, with bootstrap 95% CIs.

## Quickstart

```bash
# Steering vector (no training)
uv run python steering/extract_vector.py --layers 14 --n-pairs 150
uv run python steering/run_steering_eval.py --config steering/steering_config.yaml

# Trained gated cross-attention
uv run python gated-cross-attention/run.py --config gated-cross-attention/config.yaml
```

See `steering/README.md` and `gated-cross-attention/README.md` for details. Config: `steering/steering_config.yaml`, `gated-cross-attention/config.yaml`. Rules: `constitution.txt`.
