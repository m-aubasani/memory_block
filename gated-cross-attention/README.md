# 🛡️ Out-of-Band Alignment: Mid-Layer Axiom Injection (Gated Cross-Attention)

This folder contains the original trained architecture approach for safety alignment using mid-layer **Gated Cross-Attention** modules and a **Constraint Encoder**.

---

## Architecture Overview

1. **`model.py`**:
   - `ConstraintEncoder`: Encodes the immutable constitutional axioms (`constitution.txt`) into continuous key/value representations.
   - `GatedCrossAttention`: Injects constitutional representations into intermediate transformer hidden states with a learned gating parameter $\sigma(\text{gate})$.
   - `AlignedInjectedLLM`: Wraps a frozen base causal language model (`Qwen/Qwen2.5-1.5B-Instruct`), attaching forward hooks at target extraction/injection layers.

2. **`train.py`**:
   - Trains the gate and cross-attention parameters on contrastive safety pairs while freezing base LLM weights.
   - Applies loss masking so the model is penalized only on safe response generation, ignoring user adversarial prompts.

3. **`inference.py`**:
   - `InjectedGenerator`: Attaches forward hooks and injects constitutional memory during autoregressive generation.

4. **`run.py`**:
   - End-to-end training and evaluation pipeline configured by `config.yaml`.

---

## Running Gated Cross-Attention

To train and evaluate the Gated Cross-Attention module:

```bash
uv run python gated-cross-attention/run.py --config gated-cross-attention/config.yaml
```
