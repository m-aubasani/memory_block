# CAA Steering Vectors for Safety Alignment

This module implements high-performance inference-time safety alignment via Contrastive Activation Addition (CAA) / difference-in-means steering vectors without requiring training loops or modifying base model weights.

---

## ⚡ Key Optimizations & Speedups

1. **Generation-First Architecture**:
   - Generates all completions across the entire sweep (Baseline, System Prompt, and all Steered configurations) in one continuous pass while the base model is loaded in VRAM.
   - Then loads `GLiGuardChecker` and `RefusalChecker` **once** to batch-score all completions in bulk, eliminating repetitive model unloads and memory swaps.

2. **Length-Bucketed / Sorted Batching**:
   - Sorts prompts by sequence length during batched generation to minimize left-padding overhead, before restoring original ordering.

3. **Inference & Attention Optimizations**:
   - `attn_implementation="sdpa"` (PyTorch native Scaled Dot-Product Attention).
   - Executed under `torch.inference_mode()` with `use_cache=True`.
   - Batch size increased to `32` for generation and `64` for classification.

4. **Exploratory Sweep Mode**:
   - `max_new_tokens: 40` default for parameter sweeps (sufficient to detect refusal vs. compliance in opening tokens).

---

## 📊 Evaluation Datasets

1. **Adversarial / Attack Benchmarks**:
   - **`jailbreakbench`** (default): Real jailbreak-wrapped attack prompts from all 12
     available `JailbreakBench/artifacts` transfer artifacts (PAIR / GCG /
     prompt_with_random_search x vicuna-13b-v1.5, llama-2-7b-chat-hf,
     gpt-3.5-turbo-1106, gpt-4-0125-preview; ~1037 prompts) **plus** a static
     template-wrapping fallback subset (DAN / AIM / Developer-Mode framings over
     50 JBB goals; 200 prompts). Each prompt is tagged with its `subset`
     (method/source-model) so results are reported per attack family, not as one
     aggregate. Saved to `steering/results/per_subset_safety_jailbreakbench.csv`.
   - **Family filtering**: set `evaluation.attack_family_filter` in
     `steering_config.yaml` to a list of subset prefixes (e.g. `jbb_GCG`,
     `template_jailbreak`) to run ONLY those attack families (all their prompts,
     no cap; default top-5 ≈ 582 prompts). Set it to `null` to keep all families.
   - **`harmbench`** (CAIS HarmBench, 400 behaviors): Plain behavior text only —
     HarmBench provides no ready-made adversarial test cases without running its
     own attack pipeline, so this is effectively disabled by default.
   - **`pku`** (`PKU-Alignment/PKU-SafeRLHF` test split).

2. **Benign Over-Refusal Benchmarks**:
   - **`xstest`** (`walledai/XSTest`, 250 benign safe prompts with sensitive keywords like "kill a process").
   - **`jailbreakbench`** (benign split, 100 prompts).
   - **`pku`** (`PKU-Alignment/PKU-SafeRLHF` safe split).

---

## 🚀 Quickstart

### 1. Vector Extraction

Extract difference-in-means steering vectors:

```bash
uv run python steering/extract_vector.py --layers 7 10 14 18 21 --n-pairs 150
```

### 2. Fast Parameter Sweep (JailbreakBench + XSTest)

```bash
uv run python steering/run_steering_eval.py --config steering/steering_config.yaml
```

### 3. Run on HarmBench

```bash
uv run python steering/run_steering_eval.py --adversarial-dataset harmbench
```

### 4. Run Full Evaluation on Both Benchmarks

```bash
uv run python steering/run_steering_eval.py --eval-all
```

Outputs:
- Aggregate sweep metrics: `steering/results/sweep_results_<dataset>.csv`
- Comparative summary: `steering/results/summary_comparison_<dataset>.csv`
- **Per-prompt completions & verdicts**:
  - `steering/results/generations_adversarial_<dataset>.csv` (Columns: `Prompt`, `Baseline_Response`, `Baseline_Safe`, `SysPrompt_Response`, `SysPrompt_Safe`, `L{layer}_{mode}_{param}_Response`, `_Safe`)
  - `steering/results/generations_benign_<dataset>.csv` (Columns: `Prompt`, `Baseline_Response`, `Baseline_Refused`, `SysPrompt_Response`, `SysPrompt_Refused`, `L{layer}_{mode}_{param}_Response`, `_Refused`)
- **Live W&B Tables**:
  - `results/sweep_table`
  - `results/summary_table`
  - `generations/adversarial_<dataset>`
  - `generations/benign_<dataset>`

---

## ⚙️ Configuration (`steering_config.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  dtype: "bfloat16"
  attn_implementation: "sdpa"

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
  adversarial_dataset: "jailbreakbench" # "jailbreakbench", "harmbench", or "pku"
  benign_dataset: "xstest"              # "xstest", "jailbreakbench", or "pku"
  batch_size: 32
  classifier_batch_size: 64
  max_new_tokens: 40                    # 40 for fast sweep, 100 for full text
  constitution_path: "constitution.txt"
  eval_cache_dir: "steering/eval_cache"
  refusal_classifier_model: "natong19/refusal_classifier"
  guard_model: "fastino/gliguard-LLMGuardrails-300M"
  filter_eval_with_guard: false
  bootstrap_iterations: 1000
  bootstrap_seed: 42
  results_csv_path: "steering/results/sweep_results.csv"

wandb:
  enabled: true
  project: "memory-block-alignment"
  entity: "mr_letters-personal"
  run_name: "caa-steering-sweep"
  mode: "online"
  tags:
    - "steering-vector"
    - "caa"
    - "jailbreakbench"
    - "harmbench"
    - "xstest"
```
