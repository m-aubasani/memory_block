import os
import sys
import json
import math
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from refusal_checker import RefusalChecker
from gliguard_checker import GLiGuardChecker
from steering.steering_hook import SteeringHook
from steering.generator import SteeredGenerator
from steering.dataset_loader import get_or_create_evaluation_suite


def load_yaml_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_bootstrap_ci(scores, num_iterations=1000, seed=42, alpha=0.05):
    """
    Computes percentage mean and 95% bootstrap confidence intervals [ci_low, ci_high].
    """
    arr = np.array(scores, dtype=np.float64) * 100.0
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    n = len(arr)
    boot_means = np.empty(num_iterations, dtype=np.float64)

    for i in range(num_iterations):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    mean_val = float(np.mean(arr))
    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, ci_low, ci_high


def compute_subset_rates(scores, subset_labels, num_iterations=1000, seed=42):
    """
    Per-attack-subset mean + bootstrap CI. scores[i] corresponds to subset_labels[i].
    Returns {subset: (rate, ci_low, ci_high)} in first-appearance order.
    """
    subset_names = list(dict.fromkeys(subset_labels))
    rates = {}
    for s in subset_names:
        sub_scores = [sc for i, sc in enumerate(scores) if subset_labels[i] == s]
        rates[s] = compute_bootstrap_ci(sub_scores, num_iterations=num_iterations, seed=seed)
    return rates


def generate_in_batches_fast(
    generate_fn,
    formatted_prompts: list,
    tokenizer,
    device,
    batch_size: int = 32,
    max_new_tokens: int = 40,
    desc: str = "Generating",
):
    """
    High-performance batched text generation using length-bucketed left-padding and torch.inference_mode.
    Minimizes redundant padding tokens across samples within each batch.
    """
    if not formatted_prompts:
        return []

    # 1. Sort prompts by length to minimize left-padding overhead within batches
    indexed_prompts = list(enumerate(formatted_prompts))
    indexed_prompts.sort(key=lambda x: len(x[1]))

    responses_with_indices = []

    for i in tqdm(range(0, len(indexed_prompts), batch_size), desc=desc, leave=False):
        batch = indexed_prompts[i : i + batch_size]
        batch_indices = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            outputs = generate_fn(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        input_len = inputs.input_ids.shape[1]
        for j, orig_idx in enumerate(batch_indices):
            gen_tokens = outputs[j][input_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            responses_with_indices.append((orig_idx, decoded))

    # 2. Restore original ordering
    responses_with_indices.sort(key=lambda x: x[0])
    return [item[1] for item in responses_with_indices]


def run_sweep(config_path: str = "steering/steering_config.yaml", override_adv_dataset: str = None):
    config = load_yaml_config(config_path)
    model_cfg = config.get("model", {})
    sweep_cfg = config.get("sweep", {})
    eval_cfg = config.get("evaluation", {})
    wandb_cfg = config.get("wandb", {})

    seed = eval_cfg.get("bootstrap_seed", 42)
    set_seed(seed)

    adv_dataset_name = override_adv_dataset or eval_cfg.get("adversarial_dataset", "jailbreakbench")
    benign_dataset_name = eval_cfg.get("benign_dataset", "xstest")

    # Initialize Weights & Biases if enabled
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        import wandb
        tags = wandb_cfg.get("tags", ["steering-vector", "caa"])
        if adv_dataset_name not in tags:
            tags.append(adv_dataset_name)
        wandb.init(
            project=wandb_cfg.get("project", "memory-block-alignment"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", f"caa-steering-{adv_dataset_name}"),
            mode=wandb_cfg.get("mode", "online"),
            tags=tags,
            config=config,
        )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=======================================================")
        print(f"🎯 ACCELERATED STEERING VECTOR EVALUATION & PARAMETER SWEEP")
        print(f"=======================================================")
        print(f"Device:               {device}")
        print(f"Model:                {model_cfg.get('name')}")
        print(f"Adversarial Dataset:  {adv_dataset_name}")
        print(f"Benign Dataset:       {benign_dataset_name}")
        print(f"Max New Tokens:       {eval_cfg.get('max_new_tokens', 40)}")
        print(f"Generation Batch Size:{eval_cfg.get('batch_size', 32)}")
        print(f"Sweep Layers:         {sweep_cfg.get('layers')}")
        print(f"Sweep Modes:          {sweep_cfg.get('modes')}")
        print(f"Add Coefficients:     {sweep_cfg.get('add_coefficients')}")
        print(f"Rotate Angles (deg):  {sweep_cfg.get('rotate_angles_deg')}")
        print(f"W&B Logging:          {use_wandb}")
        print(f"=======================================================\n")

        dtype_str = model_cfg.get("dtype", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # 1. Setup Tokenizer & Base Model
        model_name = model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        attn_impl = model_cfg.get("attn_implementation", "sdpa")
        print(f"Loading base model '{model_name}' (attn_implementation='{attn_impl}')...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                attn_implementation=attn_impl,
            ).to(device)
        except Exception as e:
            print(f"Warning: could not load with attn_implementation='{attn_impl}' ({e}), falling back to default.")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
            ).to(device)
        base_model.eval()

        # Load Constitution
        constitution_path = eval_cfg.get("constitution_path", "constitution.txt")
        if not os.path.exists(constitution_path):
            constitution_path = os.path.join(PROJECT_ROOT, constitution_path)
        with open(constitution_path, "r", encoding="utf-8") as f:
            constitution = f.read()

        # 2. Obtain Fixed Evaluation Prompts
        cache_dir = eval_cfg.get("eval_cache_dir", "steering/eval_cache")
        num_samples = eval_cfg.get("num_samples", None)
        harmbench_url = eval_cfg.get("harmbench_url", "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv")
        guard_model_name = eval_cfg.get("guard_model", "fastino/gliguard-LLMGuardrails-300M")
        refusal_model_name = eval_cfg.get("refusal_classifier_model", "natong19/refusal_classifier")
        filter_eval_with_guard = eval_cfg.get("filter_eval_with_guard", False)
        batch_size = eval_cfg.get("batch_size", 32)
        clf_batch_size = eval_cfg.get("classifier_batch_size", 64)
        max_new_tokens = eval_cfg.get("max_new_tokens", 40)
        boot_iters = eval_cfg.get("bootstrap_iterations", 1000)

        adv_prompts, safe_prompts = get_or_create_evaluation_suite(
            cache_dir=cache_dir,
            adv_dataset_name=adv_dataset_name,
            benign_dataset_name=benign_dataset_name,
            harmbench_url=harmbench_url,
            num_samples=num_samples,
            seed=seed,
            tokenizer=tokenizer,
            filter_eval_with_guard=filter_eval_with_guard,
            guard_model_name=guard_model_name,
            device=str(device),
            jbb_combos=eval_cfg.get("jbb_artifact_combos"),
            template_goals=eval_cfg.get("template_goals", 50),
            attack_family_filter=eval_cfg.get("attack_family_filter"),
        )

        # Adversarial prompts carry per-prompt {prompt, subset, goal} metadata.
        adv_prompt_texts = [p["prompt"] for p in adv_prompts]
        subset_labels = [p.get("subset", "unknown") for p in adv_prompts]
        subset_names = list(dict.fromkeys(subset_labels))
        subset_counts = {s: subset_labels.count(s) for s in subset_names}
        print(f"[EVAL] Active attack-family filter: {eval_cfg.get('attack_family_filter') or 'ALL'}")
        print(f"[EVAL] Adversarial subsets: {subset_counts}")

        # Format prompts
        adv_base_formatted = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in adv_prompt_texts
        ]
        adv_sys_formatted = [
            tokenizer.apply_chat_template([{"role": "system", "content": constitution}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in adv_prompt_texts
        ]
        safe_base_formatted = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in safe_prompts
        ]
        safe_sys_formatted = [
            tokenizer.apply_chat_template([{"role": "system", "content": constitution}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in safe_prompts
        ]

        # =========================================================================
        # STAGE 1: BATCHED TEXT GENERATION (ALL CONFIGURATIONS FIRST)
        # =========================================================================
        t_gen_start = time.time()
        print("\n" + "=" * 60)
        print("⚡ STAGE 1: BATCHED TEXT GENERATION (Base Model in VRAM)")
        print("=" * 60)

        # Dictionary to hold all generated responses
        # Key format: ("baseline" | "sysprompt" | f"steered_L{l}_{m}_{param}", "adv" | "safe")
        generated_responses = {}

        print("\n[Gen 1/3] Generating Baseline (Unsteered) responses...")
        generated_responses[("baseline", "adv")] = generate_in_batches_fast(
            generate_fn=base_model.generate,
            formatted_prompts=adv_base_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Baseline Adv Gen",
        )
        generated_responses[("baseline", "safe")] = generate_in_batches_fast(
            generate_fn=base_model.generate,
            formatted_prompts=safe_base_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Baseline Safe Gen",
        )

        print("\n[Gen 2/3] Generating System Prompt responses...")
        generated_responses[("sysprompt", "adv")] = generate_in_batches_fast(
            generate_fn=base_model.generate,
            formatted_prompts=adv_sys_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="SysPrompt Adv Gen",
        )
        generated_responses[("sysprompt", "safe")] = generate_in_batches_fast(
            generate_fn=base_model.generate,
            formatted_prompts=safe_sys_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="SysPrompt Safe Gen",
        )

        # Prepare Steering Configurations
        vector_dir = "steering/vectors"
        sweep_layers = sweep_cfg.get("layers", [14])
        sweep_modes = sweep_cfg.get("modes", ["add", "rotate"])
        add_coeffs = sweep_cfg.get("add_coefficients", [0.5, 1.0, 2.0, 4.0, 8.0])
        rotate_angles = sweep_cfg.get("rotate_angles_deg", [10, 20, 30, 45, 60])

        steered_configs = []
        for layer_idx in sweep_layers:
            vector_path = os.path.join(vector_dir, f"layer_{layer_idx}.pt")
            if not os.path.exists(vector_path):
                print(f"[WARN] Vector file '{vector_path}' not found! Skipping layer {layer_idx}.")
                continue

            vec_data = torch.load(vector_path, map_location="cpu", weights_only=False)
            vector_tensor = vec_data["vector"]

            for mode in sweep_modes:
                if mode == "add":
                    for c in add_coeffs:
                        steered_configs.append({
                            "layer": layer_idx,
                            "mode": "add",
                            "coefficient": c,
                            "angle_deg": None,
                            "angle_rad": None,
                            "vector": vector_tensor,
                            "key": f"steered_L{layer_idx}_add_{c}",
                        })
                elif mode == "rotate":
                    for deg in rotate_angles:
                        rad = math.radians(deg)
                        steered_configs.append({
                            "layer": layer_idx,
                            "mode": "rotate",
                            "coefficient": None,
                            "angle_deg": deg,
                            "angle_rad": rad,
                            "vector": vector_tensor,
                            "key": f"steered_L{layer_idx}_rotate_{deg}",
                        })

        print(f"\n[Gen 3/3] Generating Steered responses ({len(steered_configs)} configurations)...")
        for cfg_item in steered_configs:
            label = f"L{cfg_item['layer']} | {cfg_item['mode']} " + (f"coeff={cfg_item['coefficient']}" if cfg_item['mode'] == "add" else f"angle={cfg_item['angle_deg']}°")
            hook = SteeringHook(
                vector=cfg_item["vector"],
                mode=cfg_item["mode"],
                coefficient=cfg_item["coefficient"] if cfg_item["coefficient"] is not None else 1.0,
                angle_rad=cfg_item["angle_rad"] if cfg_item["angle_rad"] is not None else 0.0,
            )
            generator = SteeredGenerator(base_model, {cfg_item["layer"]: hook})

            generated_responses[(cfg_item["key"], "adv")] = generate_in_batches_fast(
                generate_fn=generator.generate,
                formatted_prompts=adv_base_formatted,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                desc=f"Steered Adv ({label})",
            )
            generated_responses[(cfg_item["key"], "safe")] = generate_in_batches_fast(
                generate_fn=generator.generate,
                formatted_prompts=safe_base_formatted,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                desc=f"Steered Safe ({label})",
            )

        t_gen_end = time.time()
        print(f"✅ Text generation completed in {t_gen_end - t_gen_start:.1f}s.")

        # Clean VRAM cache before classification
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =========================================================================
        # STAGE 2: CONTINUOUS BATCH CLASSIFICATION (LOAD CLASSIFIERS ONCE)
        # =========================================================================
        t_clf_start = time.time()
        print("\n" + "=" * 60)
        print("🔍 STAGE 2: CONTINUOUS BATCH CLASSIFICATION (Single Load Pass)")
        print("=" * 60)

        # 2.1 Adversarial Safety Classification with GLiGuard (Single Pass)
        print(f"\nLoading GLiGuard classifier '{guard_model_name}' once for all adversarial responses...")
        guard = GLiGuardChecker(model_name=guard_model_name, device=str(device))

        adv_keys = ["baseline", "sysprompt"] + [cfg["key"] for cfg in steered_configs]
        adv_scores_dict = {}

        for key in tqdm(adv_keys, desc="GLiGuard Safety Scoring"):
            responses = generated_responses[(key, "adv")]
            is_safe_list = guard.is_response_safe(responses, prompts=adv_prompt_texts, batch_size=clf_batch_size)
            adv_scores_dict[key] = is_safe_list

        del guard
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2.2 Benign Over-Refusal Classification with RefusalChecker (Single Pass)
        print(f"\nLoading RefusalChecker classifier '{refusal_model_name}' once for all benign responses...")
        checker = RefusalChecker(model_name=refusal_model_name, device=device)

        safe_keys = ["baseline", "sysprompt"] + [cfg["key"] for cfg in steered_configs]
        safe_scores_dict = {}

        for key in tqdm(safe_keys, desc="RefusalChecker Scoring"):
            responses = generated_responses[(key, "safe")]
            is_refusal_list = checker.is_refusal(responses)
            safe_scores_dict[key] = is_refusal_list

        del checker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_clf_end = time.time()
        print(f"✅ Safety classification completed in {t_clf_end - t_clf_start:.1f}s.")

        # =========================================================================
        # STAGE 3: METRICS, BOOTSTRAP CONFIDENCE INTERVALS, & W&B LOGGING
        # =========================================================================
        print("\n" + "=" * 60)
        print("📊 STAGE 3: COMPUTING BOOTSTRAP CIs & SUMMARY METRICS")
        print("=" * 60)

        # Baseline stats
        base_adv_safe = adv_scores_dict["baseline"]
        base_benign_refused = safe_scores_dict["baseline"]
        base_adv_rate, base_adv_low, base_adv_high = compute_bootstrap_ci(base_adv_safe, num_iterations=boot_iters, seed=seed)
        base_ref_rate, base_ref_low, base_ref_high = compute_bootstrap_ci(base_benign_refused, num_iterations=boot_iters, seed=seed)
        base_sub_rates = compute_subset_rates(base_adv_safe, subset_labels, num_iterations=boot_iters, seed=seed)

        # SysPrompt stats
        sys_adv_safe = adv_scores_dict["sysprompt"]
        sys_benign_refused = safe_scores_dict["sysprompt"]
        sys_adv_rate, sys_adv_low, sys_adv_high = compute_bootstrap_ci(sys_adv_safe, num_iterations=boot_iters, seed=seed)
        sys_ref_rate, sys_ref_low, sys_ref_high = compute_bootstrap_ci(sys_benign_refused, num_iterations=boot_iters, seed=seed)
        sys_sub_rates = compute_subset_rates(sys_adv_safe, subset_labels, num_iterations=boot_iters, seed=seed)

        print(f"📊 Baseline Results:    Adv Safety = {base_adv_rate:.1f}% [{base_adv_low:.1f}, {base_adv_high:.1f}] | Benign Refusal = {base_ref_rate:.1f}% [{base_ref_low:.1f}, {base_ref_high:.1f}]")
        print(f"📊 SysPrompt Results:   Adv Safety = {sys_adv_rate:.1f}% [{sys_adv_low:.1f}, {sys_adv_high:.1f}] | Benign Refusal = {sys_ref_rate:.1f}% [{sys_ref_low:.1f}, {sys_ref_high:.1f}]\n")

        # Log Baseline & SysPrompt metrics to W&B
        if use_wandb and wandb.run is not None:
            wandb.log({
                "baseline/adv_safety_rate": base_adv_rate,
                "baseline/adv_safety_ci_low": base_adv_low,
                "baseline/adv_safety_ci_high": base_adv_high,
                "baseline/benign_refusal_rate": base_ref_rate,
                "baseline/benign_refusal_ci_low": base_ref_low,
                "baseline/benign_refusal_ci_high": base_ref_high,
                "sysprompt/adv_safety_rate": sys_adv_rate,
                "sysprompt/adv_safety_ci_low": sys_adv_low,
                "sysprompt/adv_safety_ci_high": sys_adv_high,
                "sysprompt/benign_refusal_rate": sys_ref_rate,
                "sysprompt/benign_refusal_ci_low": sys_ref_low,
                "sysprompt/benign_refusal_ci_high": sys_ref_high,
            })
            wandb.run.summary["baseline_adv_safety_rate"] = base_adv_rate
            wandb.run.summary["baseline_benign_refusal_rate"] = base_ref_rate
            wandb.run.summary["sysprompt_adv_safety_rate"] = sys_adv_rate
            wandb.run.summary["sysprompt_benign_refusal_rate"] = sys_ref_rate

        results = []
        for step_idx, cfg_item in enumerate(steered_configs, 1):
            key = cfg_item["key"]
            adv_safe_scores = adv_scores_dict[key]
            safe_refusal_scores = safe_scores_dict[key]

            adv_rate, adv_low, adv_high = compute_bootstrap_ci(adv_safe_scores, num_iterations=boot_iters, seed=seed)
            ref_rate, ref_low, ref_high = compute_bootstrap_ci(safe_refusal_scores, num_iterations=boot_iters, seed=seed)
            sub_rates = compute_subset_rates(adv_safe_scores, subset_labels, num_iterations=boot_iters, seed=seed)

            delta_adv_base = adv_rate - base_adv_rate
            delta_ref_base = ref_rate - base_ref_rate

            m = cfg_item["mode"]
            coeff = cfg_item["coefficient"]
            deg = cfg_item["angle_deg"]
            rad = cfg_item["angle_rad"]

            label = f"L{cfg_item['layer']} | {m} " + (f"coeff={coeff}" if m == "add" else f"angle={deg}°")
            print(f">>> Config [{step_idx:02d}]: {label:<30} | Adv Safety: {adv_rate:5.1f}% [{adv_low:4.1f}, {adv_high:4.1f}] | Benign Refusal: {ref_rate:5.1f}% [{ref_low:4.1f}, {ref_high:4.1f}]")

            row_data = {
                "layer": cfg_item["layer"],
                "mode": m,
                "coefficient": coeff,
                "angle_deg": deg,
                "angle_rad": round(rad, 4) if rad is not None else None,
                "adv_safety_rate": adv_rate,
                "adv_safety_ci_low": adv_low,
                "adv_safety_ci_high": adv_high,
                "benign_refusal_rate": ref_rate,
                "benign_refusal_ci_low": ref_low,
                "benign_refusal_ci_high": ref_high,
                "n_eval_samples": len(adv_prompts),
                "adversarial_dataset": adv_dataset_name,
                "benign_dataset": benign_dataset_name,
            }
            for s in subset_names:
                row_data[f"adv_safety_rate_{s}"] = sub_rates[s][0]
                row_data[f"n_eval_{s}"] = subset_counts[s]
            results.append(row_data)

            # Log step metrics to W&B
            if use_wandb and wandb.run is not None:
                param_val = coeff if m == "add" else deg
                wandb.log({
                    "sweep/step": step_idx,
                    "sweep/layer": cfg_item["layer"],
                    "sweep/adv_safety_rate": adv_rate,
                    "sweep/adv_safety_ci_low": adv_low,
                    "sweep/adv_safety_ci_high": adv_high,
                    "sweep/benign_refusal_rate": ref_rate,
                    "sweep/benign_refusal_ci_low": ref_low,
                    "sweep/benign_refusal_ci_high": ref_high,
                    "sweep/delta_adv_vs_baseline": delta_adv_base,
                    "sweep/delta_refusal_vs_baseline": delta_ref_base,
                    f"layer_{cfg_item['layer']}_{m}/adv_safety_rate": adv_rate,
                    f"layer_{cfg_item['layer']}_{m}/benign_refusal_rate": ref_rate,
                    f"layer_{cfg_item['layer']}_{m}/param_value": param_val,
                    **{f"sweep/subset_{s}_adv_safety_rate": sub_rates[s][0] for s in subset_names},
                })

        # Save CSV results
        results_dir = "steering/results"
        os.makedirs(results_dir, exist_ok=True)
        results_csv_path = eval_cfg.get("results_csv_path", os.path.join(results_dir, f"sweep_results_{adv_dataset_name}.csv"))

        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv_path, index=False)
        print(f"\n📁 Saved sweep results to '{results_csv_path}'")

        # Save summary table
        summary_rows = [
            {
                "method": "Baseline (Unsteered)",
                "layer": "-",
                "mode": "-",
                "parameter": "-",
                "adv_safety_rate": float(base_adv_rate),
                "adv_safety_ci": f"[{base_adv_low:.1f}, {base_adv_high:.1f}]",
                "benign_refusal_rate": float(base_ref_rate),
                "benign_refusal_ci": f"[{base_ref_low:.1f}, {base_ref_high:.1f}]",
            },
            {
                "method": "System Prompt",
                "layer": "-",
                "mode": "-",
                "parameter": "-",
                "adv_safety_rate": float(sys_adv_rate),
                "adv_safety_ci": f"[{sys_adv_low:.1f}, {sys_adv_high:.1f}]",
                "benign_refusal_rate": float(sys_ref_rate),
                "benign_refusal_ci": f"[{sys_ref_low:.1f}, {sys_ref_high:.1f}]",
            },
        ]
        for s in subset_names:
            summary_rows[0][f"adv_safety_rate_{s}"] = base_sub_rates[s][0]
            summary_rows[1][f"adv_safety_rate_{s}"] = sys_sub_rates[s][0]

        for r in results:
            param_str = f"coeff={r['coefficient']}" if r["mode"] == "add" else f"angle={r['angle_deg']}°"
            summary_row = {
                "method": f"Steering (L{r['layer']})",
                "layer": str(r["layer"]),
                "mode": str(r["mode"]),
                "parameter": str(param_str),
                "adv_safety_rate": float(r["adv_safety_rate"]),
                "adv_safety_ci": f"[{r['adv_safety_ci_low']:.1f}, {r['adv_safety_ci_high']:.1f}]",
                "benign_refusal_rate": float(r["benign_refusal_rate"]),
                "benign_refusal_ci": f"[{r['benign_refusal_ci_low']:.1f}, {r['benign_refusal_ci_high']:.1f}]",
            }
            for s in subset_names:
                summary_row[f"adv_safety_rate_{s}"] = r[f"adv_safety_rate_{s}"]
            summary_rows.append(summary_row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary_sorted = pd.concat([
            df_summary.iloc[:2],
            df_summary.iloc[2:].sort_values(by="adv_safety_rate", ascending=False)
        ], ignore_index=True)

        summary_csv_path = os.path.join(results_dir, f"summary_comparison_{adv_dataset_name}.csv")
        df_summary_sorted.to_csv(summary_csv_path, index=False)

        print("\n" + "=" * 85)
        print(f"🏆 FINAL COMPARISON SUMMARY TABLE [{adv_dataset_name.upper()} + {benign_dataset_name.upper()}]")
        print("=" * 85)
        print(df_summary_sorted.to_string(index=False))
        print("=" * 85 + "\n")

        # Per-attack-subset breakdown: baseline vs sysprompt vs top steered configs.
        matrix_rows = []
        for s in subset_names:
            matrix_rows.append({
                "attack_subset": s,
                "n": subset_counts[s],
                "Baseline": base_sub_rates[s][0],
                "SysPrompt": sys_sub_rates[s][0],
            })
        for r in sorted(results, key=lambda x: x["adv_safety_rate"], reverse=True)[:3]:
            param_str = f"coeff={r['coefficient']}" if r["mode"] == "add" else f"angle={r['angle_deg']}°"
            col = f"Steering L{r['layer']} {r['mode']} {param_str}"
            for mrow in matrix_rows:
                mrow[col] = r[f"adv_safety_rate_{mrow['attack_subset']}"]

        df_subset_matrix = pd.DataFrame(matrix_rows)
        subset_matrix_csv_path = os.path.join(results_dir, f"per_subset_safety_{adv_dataset_name}.csv")
        df_subset_matrix.to_csv(subset_matrix_csv_path, index=False)

        print("\n" + "=" * 85)
        print("🎯 PER-ATTACK-SUBSET ADVERSARIAL SAFETY RATES (% refused-safe)")
        print(f"    (a) JBB transfer by method/source-model, (b) template-wrapping, (c) benign XSTest refusal is in the table above")
        print("=" * 85)
        print(df_subset_matrix.round(1).to_string(index=False))
        print("=" * 85 + "\n")
        print(f"📁 Saved per-attack-subset safety matrix to '{subset_matrix_csv_path}'")

        # =========================================================================
        # STAGE 4: BUILD PER-PROMPT GENERATIONS DATAFRAMES & UPLOAD
        # =========================================================================
        print("\n" + "=" * 60)
        print("💾 STAGE 4: SAVING PROMPT & COMPLETION COMPARISONS")
        print("=" * 60)

        # 4.1 Adversarial Generations Table
        adv_gen_dict = {
            "Subset": subset_labels,
            "Goal": [p.get("goal") for p in adv_prompts],
            "Prompt": adv_prompt_texts,
            "Baseline_Response": generated_responses[("baseline", "adv")],
            "Baseline_Safe": adv_scores_dict["baseline"],
            "SysPrompt_Response": generated_responses[("sysprompt", "adv")],
            "SysPrompt_Safe": adv_scores_dict["sysprompt"],
        }
        for cfg in steered_configs:
            label_col = f"L{cfg['layer']}_{cfg['mode']}_" + (f"coeff_{cfg['coefficient']}" if cfg['mode'] == "add" else f"angle_{cfg['angle_deg']}deg")
            adv_gen_dict[f"{label_col}_Response"] = generated_responses[(cfg["key"], "adv")]
            adv_gen_dict[f"{label_col}_Safe"] = adv_scores_dict[cfg["key"]]

        df_adv_generations = pd.DataFrame(adv_gen_dict)
        adv_gen_csv_path = os.path.join(results_dir, f"generations_adversarial_{adv_dataset_name}.csv")
        df_adv_generations.to_csv(adv_gen_csv_path, index=False)
        print(f"📁 Saved adversarial prompt/response comparisons to '{adv_gen_csv_path}'")

        # 4.2 Safe/Benign Generations Table
        safe_gen_dict = {
            "Prompt": safe_prompts,
            "Baseline_Response": generated_responses[("baseline", "safe")],
            "Baseline_Refused": safe_scores_dict["baseline"],
            "SysPrompt_Response": generated_responses[("sysprompt", "safe")],
            "SysPrompt_Refused": safe_scores_dict["sysprompt"],
        }
        for cfg in steered_configs:
            label_col = f"L{cfg['layer']}_{cfg['mode']}_" + (f"coeff_{cfg['coefficient']}" if cfg['mode'] == "add" else f"angle_{cfg['angle_deg']}deg")
            safe_gen_dict[f"{label_col}_Response"] = generated_responses[(cfg["key"], "safe")]
            safe_gen_dict[f"{label_col}_Refused"] = safe_scores_dict[cfg["key"]]

        df_safe_generations = pd.DataFrame(safe_gen_dict)
        safe_gen_csv_path = os.path.join(results_dir, f"generations_benign_{benign_dataset_name}.csv")
        df_safe_generations.to_csv(safe_gen_csv_path, index=False)
        print(f"📁 Saved benign prompt/response comparisons to '{safe_gen_csv_path}'")

        # Log Final Tables, Artifacts, and Best Summary to W&B
        if use_wandb and wandb.run is not None:
            wandb.log({
                "results/sweep_table": wandb.Table(dataframe=df_results),
                "results/summary_table": wandb.Table(dataframe=df_summary_sorted),
                f"results/per_subset_safety_{adv_dataset_name}": wandb.Table(dataframe=df_subset_matrix),
                f"generations/adversarial_{adv_dataset_name}": wandb.Table(dataframe=df_adv_generations),
                f"generations/benign_{benign_dataset_name}": wandb.Table(dataframe=df_safe_generations),
            })

            wandb.save(results_csv_path, base_path=os.path.dirname(results_csv_path))
            wandb.save(summary_csv_path, base_path=os.path.dirname(summary_csv_path))
            wandb.save(subset_matrix_csv_path, base_path=os.path.dirname(subset_matrix_csv_path))
            wandb.save(adv_gen_csv_path, base_path=os.path.dirname(adv_gen_csv_path))
            wandb.save(safe_gen_csv_path, base_path=os.path.dirname(safe_gen_csv_path))

            if len(results) > 0:
                best_steered = max(results, key=lambda x: x["adv_safety_rate"])
                best_param = f"coeff={best_steered['coefficient']}" if best_steered["mode"] == "add" else f"angle={best_steered['angle_deg']}°"
                wandb.run.summary["best_steered_adv_safety_rate"] = best_steered["adv_safety_rate"]
                wandb.run.summary["best_steered_benign_refusal_rate"] = best_steered["benign_refusal_rate"]
                wandb.run.summary["best_steered_config"] = f"L{best_steered['layer']} {best_steered['mode']} {best_param}"

    finally:
        if use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run High-Speed Steering Vector Sweep Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="steering/steering_config.yaml",
        help="Path to steering config YAML",
    )
    parser.add_argument(
        "--adversarial-dataset",
        type=str,
        default=None,
        help="Adversarial dataset: 'jailbreakbench', 'harmbench', or 'pku'",
    )
    parser.add_argument(
        "--benign-dataset",
        type=str,
        default=None,
        help="Benign dataset: 'xstest', 'jailbreakbench', or 'pku'",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Run sweep across both JailbreakBench and HarmBench sequentially",
    )
    args = parser.parse_args()

    if args.eval_all:
        print("\n>>> Running evaluation on JailbreakBench...")
        run_sweep(config_path=args.config, override_adv_dataset="jailbreakbench")
        print("\n>>> Running evaluation on HarmBench...")
        run_sweep(config_path=args.config, override_adv_dataset="harmbench")
    else:
        run_sweep(config_path=args.config, override_adv_dataset=args.adversarial_dataset)


if __name__ == "__main__":
    main()
