# CAA Steering Vectors for Safety Alignment

This module implements inference-time safety alignment via Contrastive Activation Addition (CAA) / difference-in-means steering vectors without requiring training loops or modifying base model weights.

---

## Key Features

1. **Difference-in-Means Vector Extraction (`extract_vector.py`)**:
   - Extracts layer-wise mean activation differences between safe and unsafe responses:
     $$\vec{v} = \mathbb{E}[h_{\text{safe}}] - \mathbb{E}[h_{\text{unsafe}}]$$
   - Extracts at the last token position using full chat-templated conversations.
   - Vectors point *toward* safety.

2. **Interchangeable Steering Modes (`steering_hook.py`)**:
   - `add`: Plain CAA additive steering:
     $$h_{\text{new}} = h + \text{coefficient} \cdot \vec{v}$$
   - `rotate`: Norm-preserving spherical interpolation:
     $$\hat{h} = \frac{h}{\|h\|_2}, \quad \hat{v} = \frac{\vec{v}}{\|\vec{v}\|_2}$$
     $$h_{\text{rot}} = \hat{h} \cos(\theta) + \hat{v} \sin(\theta)$$
     $$h_{\text{new}} = h_{\text{rot}} \cdot \|h\|_2$$
   - **Numerical Precision**: Internal arithmetic is executed in `float32` before casting back to the model's native dtype (`bfloat16`) to prevent precision drift.

3. **Fixed Evaluation Set Caching & Efficiency**:
   - Evaluates on a fixed prompt set cached at `steering/eval_cache/fixed_eval_set.json`.
   - **Caching Approach**: We implemented a `steering/`-local evaluation loop (`run_steering_eval.py`) that loads from / populates the fixed JSON cache and calculates Baseline and System-Prompt responses **once**, sharing them across all sweep parameter configurations.
   - Incorporates **1,000-iteration Bootstrap 95% Confidence Intervals** for both adversarial safety rates and benign over-refusal rates.

4. **Weights & Biases (W&B) Logging**:
   - Logs sweep progression, per-configuration safety & refusal rates, confidence intervals, and comparison deltas vs. Baseline and System Prompt.
   - Automatically logs interactive `wandb.Table`s for `sweep_results.csv` and `summary_comparison.csv`.

---

## Quickstart

### 1. Vector Extraction

Extract steering vectors for target layers (e.g. 7, 10, 14, 18, 21):

```bash
uv run python steering/extract_vector.py --layers 7 10 14 18 21 --n-pairs 150
```

Vectors are saved to `steering/vectors/layer_{L}.pt` along with `steering/vectors/metadata.json`.

### 2. Parameter Sweep & Evaluation (with W&B)

Run the evaluation sweep across modes (`add`, `rotate`) and parameter values on layer 14:

```bash
uv run python steering/run_steering_eval.py --config steering/steering_config.yaml
```

Outputs:
- Detailed metrics & CIs: `steering/results/sweep_results.csv`
- Comparative summary: `steering/results/summary_comparison.csv`
- W&B dashboard live updates with tables and metric curves.

---

## Configuration (`steering_config.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  dtype: "bfloat16"

extraction:
  layers: [7, 10, 14, 18, 21]
  n_pairs: 150
  dataset_name: "PKU-Alignment/PKU-SafeRLHF"
  seed: 42

sweep:
  layers: [14]
  modes: ["add", "rotate"]
  add_coefficients: [0.5, 1.0, 2.0, 4.0, 8.0]
  rotate_angles_deg: [10, 20, 30, 45, 60]

evaluation:
  num_samples: 300
  batch_size: 16
  max_new_tokens: 100
  eval_cache_path: "steering/eval_cache/fixed_eval_set.json"
  refusal_classifier_model: "natong19/refusal_classifier"
  guard_model: "fastino/gliguard-LLMGuardrails-300M"
  bootstrap_iterations: 1000
  bootstrap_seed: 42

wandb:
  enabled: true
  project: "memory-block-alignment"
  entity: "mr_letters-personal"
  run_name: "caa-steering-sweep"
  mode: "online"             # options: "online", "offline", "disabled"
  tags:
    - "steering-vector"
    - "caa"
    - "ai-alignment"
```
