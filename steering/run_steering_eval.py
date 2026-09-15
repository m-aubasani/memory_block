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

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_loader import AlignmentDataset
from refusal_checker import RefusalChecker
from gliguard_checker import GLiGuardChecker
from steering.steering_hook import SteeringHook
from steering.generator import SteeredGenerator
from eval import generate_in_batches


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


def get_or_create_cached_eval_prompts(
    cache_path: str,
    tokenizer,
    num_samples: int = 300,
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    guard_model_name: str = "fastino/gliguard-LLMGuardrails-300M",
    filter_eval_with_guard: bool = True,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Loads fixed evaluation prompts from cache if available, or derives and saves them.
    Ensures identical evaluation prompts across all sweep iterations.
    """
    if os.path.exists(cache_path):
        print(f"\n📂 Loading cached evaluation prompts from '{cache_path}'...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            adv_prompts = cached_data["adversarial"]
            safe_prompts = cached_data["safe"]
            print(f"Loaded {len(adv_prompts)} adversarial and {len(safe_prompts)} safe/benign cached prompts.")
            return adv_prompts, safe_prompts

    print(f"\n⚙️ Generating fixed evaluation prompt cache ({num_samples} samples per split)...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Adversarial prompts (both responses unsafe)
    print("Deriving adversarial evaluation prompts...")
    adv_dataset = AlignmentDataset(
        tokenizer=tokenizer,
        split="test",
        max_samples=num_samples,
        dataset_name=dataset_name,
        eval_mode="adversarial",
        filter_eval_with_guard=filter_eval_with_guard,
        guard_model_name=guard_model_name,
        seed=seed,
        device=device,
    )
    adv_prompts = [item["prompt"] for item in adv_dataset.dataset]

    # 2. Safe/benign prompts (both responses safe)
    print("Deriving safe/benign evaluation prompts...")
    safe_dataset = AlignmentDataset(
        tokenizer=tokenizer,
        split="test",
        max_samples=num_samples,
        dataset_name=dataset_name,
        eval_mode="safe",
        filter_eval_with_guard=filter_eval_with_guard,
        guard_model_name=guard_model_name,
        seed=seed,
        device=device,
    )
    safe_prompts = [item["prompt"] for item in safe_dataset.dataset]

    cache_data = {
        "adversarial": adv_prompts,
        "safe": safe_prompts,
        "metadata": {
            "num_samples": num_samples,
            "dataset_name": dataset_name,
            "seed": seed,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)
    print(f"✅ Cached {len(adv_prompts)} adversarial and {len(safe_prompts)} safe prompts to '{cache_path}'.\n")

    return adv_prompts, safe_prompts


def run_sweep(config_path: str = "steering/steering_config.yaml"):
    config = load_yaml_config(config_path)
    model_cfg = config.get("model", {})
    sweep_cfg = config.get("sweep", {})
    eval_cfg = config.get("evaluation", {})
    wandb_cfg = config.get("wandb", {})

    seed = eval_cfg.get("bootstrap_seed", 42)
    set_seed(seed)

    # Initialize Weights & Biases if enabled
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "memory-block-alignment"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", "caa-steering-sweep"),
            mode=wandb_cfg.get("mode", "online"),
            tags=wandb_cfg.get("tags", ["steering-vector", "caa"]),
            config=config,
        )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=======================================================")
        print(f"🎯 STEERING VECTOR EVALUATION & PARAMETER SWEEP")
        print(f"=======================================================")
        print(f"Device:               {device}")
        print(f"Model:                {model_cfg.get('name')}")
        print(f"Eval Samples:         {eval_cfg.get('num_samples')}")
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

        print(f"Loading base model '{model_name}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
        ).to(device)
        base_model.eval()

        # Load Constitution
        constitution_path = eval_cfg.get("constitution_path", "constitution.txt")
        with open(constitution_path, "r", encoding="utf-8") as f:
            constitution = f.read()

        # 2. Obtain Fixed Evaluation Prompts
        eval_cache_path = eval_cfg.get("eval_cache_path", "steering/eval_cache/fixed_eval_set.json")
        num_samples = eval_cfg.get("num_samples", 300)
        guard_model_name = eval_cfg.get("guard_model", "fastino/gliguard-LLMGuardrails-300M")
        refusal_model_name = eval_cfg.get("refusal_classifier_model", "natong19/refusal_classifier")
        dataset_name = config.get("extraction", {}).get("dataset_name", "PKU-Alignment/PKU-SafeRLHF")
        filter_eval_with_guard = eval_cfg.get("filter_eval_with_guard", True)
        batch_size = eval_cfg.get("batch_size", 16)
        max_new_tokens = eval_cfg.get("max_new_tokens", 100)
        boot_iters = eval_cfg.get("bootstrap_iterations", 1000)

        adv_prompts, safe_prompts = get_or_create_cached_eval_prompts(
            cache_path=eval_cache_path,
            tokenizer=tokenizer,
            num_samples=num_samples,
            dataset_name=dataset_name,
            guard_model_name=guard_model_name,
            filter_eval_with_guard=filter_eval_with_guard,
            seed=seed,
            device=device,
        )

        # Configure tokenizer for left-padding during batch generation
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        # Format prompts
        adv_base_formatted = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in adv_prompts
        ]
        adv_sys_formatted = [
            tokenizer.apply_chat_template([{"role": "system", "content": constitution}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in adv_prompts
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
        # STEP 1: Compute Baseline & System Prompt Evaluations (ONCE)
        # =========================================================================
        print("\n--- Generating & Evaluating Baseline (Unsteered) Responses ---")
        base_adv_responses = generate_in_batches(
            generate_fn=base_model.generate,
            formatted_prompts=adv_base_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Baseline Adv Gen",
        )
        base_safe_responses = generate_in_batches(
            generate_fn=base_model.generate,
            formatted_prompts=safe_base_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Baseline Safe Gen",
        )

        print("\n--- Generating & Evaluating System Prompt Responses ---")
        sys_adv_responses = generate_in_batches(
            generate_fn=base_model.generate,
            formatted_prompts=adv_sys_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="SysPrompt Adv Gen",
        )
        sys_safe_responses = generate_in_batches(
            generate_fn=base_model.generate,
            formatted_prompts=safe_sys_formatted,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="SysPrompt Safe Gen",
        )

        # Evaluate Baseline & SysPrompt Safety
        print("\nScoring Baseline and System Prompt responses...")
        guard = GLiGuardChecker(model_name=guard_model_name, device=device)
        base_adv_safe = guard.is_response_safe(base_adv_responses, prompts=adv_prompts, batch_size=batch_size)
        sys_adv_safe = guard.is_response_safe(sys_adv_responses, prompts=adv_prompts, batch_size=batch_size)
        del guard
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        checker = RefusalChecker(model_name=refusal_model_name, device=device)
        base_benign_refused = checker.is_refusal(base_safe_responses)
        sys_benign_refused = checker.is_refusal(sys_safe_responses)
        del checker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_adv_rate, base_adv_low, base_adv_high = compute_bootstrap_ci(base_adv_safe, num_iterations=boot_iters, seed=seed)
        base_ref_rate, base_ref_low, base_ref_high = compute_bootstrap_ci(base_benign_refused, num_iterations=boot_iters, seed=seed)

        sys_adv_rate, sys_adv_low, sys_adv_high = compute_bootstrap_ci(sys_adv_safe, num_iterations=boot_iters, seed=seed)
        sys_ref_rate, sys_ref_low, sys_ref_high = compute_bootstrap_ci(sys_benign_refused, num_iterations=boot_iters, seed=seed)

        print(f"\n📊 Baseline Results:    Adv Safety = {base_adv_rate:.1f}% [{base_adv_low:.1f}, {base_adv_high:.1f}] | Benign Refusal = {base_ref_rate:.1f}% [{base_ref_low:.1f}, {base_ref_high:.1f}]")
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

        # =========================================================================
        # STEP 2: Parameter Sweep Over Steering Vectors
        # =========================================================================
        results = []
        vector_dir = "steering/vectors"

        sweep_layers = sweep_cfg.get("layers", [14])
        sweep_modes = sweep_cfg.get("modes", ["add", "rotate"])
        add_coeffs = sweep_cfg.get("add_coefficients", [0.5, 1.0, 2.0, 4.0, 8.0])
        rotate_angles = sweep_cfg.get("rotate_angles_deg", [10, 20, 30, 45, 60])

        step_idx = 0
        for layer_idx in sweep_layers:
            vector_path = os.path.join(vector_dir, f"layer_{layer_idx}.pt")
            if not os.path.exists(vector_path):
                print(f"⚠️ Vector file '{vector_path}' not found! Skipping layer {layer_idx}.")
                continue

            vec_data = torch.load(vector_path, map_location="cpu", weights_only=False)
            vector_tensor = vec_data["vector"]
            print(f"\n==========================================")
            print(f"⚙️ Evaluating Layer {layer_idx} (Vector norm = {vec_data['norm']:.4f})")
            print(f"==========================================")

            for mode in sweep_modes:
                if mode == "add":
                    param_list = [("add", c, None, None) for c in add_coeffs]
                elif mode == "rotate":
                    param_list = [("rotate", None, deg, math.radians(deg)) for deg in rotate_angles]
                else:
                    continue

                for m, coeff, deg, rad in param_list:
                    step_idx += 1
                    label = f"mode={m}, coeff={coeff}" if m == "add" else f"mode={m}, angle={deg}° ({rad:.3f} rad)"
                    print(f"\n>>> Running configuration [{step_idx}]: Layer {layer_idx} | {label}")

                    hook = SteeringHook(
                        vector=vector_tensor,
                        mode=m,
                        coefficient=coeff if coeff is not None else 1.0,
                        angle_rad=rad if rad is not None else 0.0,
                    )
                    generator = SteeredGenerator(base_model, {layer_idx: hook})

                    # Generate responses
                    steered_adv_responses = generate_in_batches(
                        generate_fn=generator.generate,
                        formatted_prompts=adv_base_formatted,
                        tokenizer=tokenizer,
                        device=device,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        desc=f"Steered Adv Gen ({label})",
                    )
                    steered_safe_responses = generate_in_batches(
                        generate_fn=generator.generate,
                        formatted_prompts=safe_base_formatted,
                        tokenizer=tokenizer,
                        device=device,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        desc=f"Steered Safe Gen ({label})",
                    )

                    # Score responses
                    guard = GLiGuardChecker(model_name=guard_model_name, device=device)
                    steered_adv_safe = guard.is_response_safe(steered_adv_responses, prompts=adv_prompts, batch_size=batch_size)
                    del guard
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    checker = RefusalChecker(model_name=refusal_model_name, device=device)
                    steered_safe_refused = checker.is_refusal(steered_safe_responses)
                    del checker
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    adv_rate, adv_low, adv_high = compute_bootstrap_ci(steered_adv_safe, num_iterations=boot_iters, seed=seed)
                    ref_rate, ref_low, ref_high = compute_bootstrap_ci(steered_safe_refused, num_iterations=boot_iters, seed=seed)

                    delta_adv_base = adv_rate - base_adv_rate
                    delta_ref_base = ref_rate - base_ref_rate

                    print(f"  ↳ Adv Safety:     {adv_rate:.1f}% [{adv_low:.1f}, {adv_high:.1f}] (Delta vs Base: {delta_adv_base:+.1f}%)")
                    print(f"  ↳ Benign Refusal: {ref_rate:.1f}% [{ref_low:.1f}, {ref_high:.1f}] (Delta vs Base: {delta_ref_base:+.1f}%)")

                    row_data = {
                        "layer": layer_idx,
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
                    }
                    results.append(row_data)

                    # Log per-iteration sweep metrics to W&B
                    if use_wandb and wandb.run is not None:
                        param_val = coeff if m == "add" else deg
                        wandb.log({
                            "sweep/step": step_idx,
                            "sweep/layer": layer_idx,
                            "sweep/adv_safety_rate": adv_rate,
                            "sweep/adv_safety_ci_low": adv_low,
                            "sweep/adv_safety_ci_high": adv_high,
                            "sweep/benign_refusal_rate": ref_rate,
                            "sweep/benign_refusal_ci_low": ref_low,
                            "sweep/benign_refusal_ci_high": ref_high,
                            "sweep/delta_adv_vs_baseline": delta_adv_base,
                            "sweep/delta_refusal_vs_baseline": delta_ref_base,
                            f"layer_{layer_idx}_{m}/adv_safety_rate": adv_rate,
                            f"layer_{layer_idx}_{m}/benign_refusal_rate": ref_rate,
                            f"layer_{layer_idx}_{m}/param_value": param_val,
                        })

        # Restore tokenizer padding side
        tokenizer.padding_side = orig_padding_side

        # =========================================================================
        # STEP 3: Save Results & Summary
        # =========================================================================
        results_dir = "steering/results"
        os.makedirs(results_dir, exist_ok=True)
        results_csv_path = eval_cfg.get("results_csv_path", os.path.join(results_dir, "sweep_results.csv"))

        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv_path, index=False)
        print(f"\n📁 Saved sweep results to '{results_csv_path}'")

        # Save summary table comparing Baseline, SysPrompt, and Steering Configurations
        summary_rows = [
            {
                "method": "Baseline (Unsteered)",
                "layer": "-",
                "mode": "-",
                "parameter": "-",
                "adv_safety_rate": base_adv_rate,
                "adv_safety_ci": f"[{base_adv_low:.1f}, {base_adv_high:.1f}]",
                "benign_refusal_rate": base_ref_rate,
                "benign_refusal_ci": f"[{base_ref_low:.1f}, {base_ref_high:.1f}]",
            },
            {
                "method": "System Prompt",
                "layer": "-",
                "mode": "-",
                "parameter": "-",
                "adv_safety_rate": sys_adv_rate,
                "adv_safety_ci": f"[{sys_adv_low:.1f}, {sys_adv_high:.1f}]",
                "benign_refusal_rate": sys_ref_rate,
                "benign_refusal_ci": f"[{sys_ref_low:.1f}, {sys_ref_high:.1f}]",
            },
        ]

        for r in results:
            param_str = f"coeff={r['coefficient']}" if r["mode"] == "add" else f"angle={r['angle_deg']}°"
            summary_rows.append({
                "method": f"Steering (L{r['layer']})",
                "layer": r["layer"],
                "mode": r["mode"],
                "parameter": param_str,
                "adv_safety_rate": r["adv_safety_rate"],
                "adv_safety_ci": f"[{r['adv_safety_ci_low']:.1f}, {r['adv_safety_ci_high']:.1f}]",
                "benign_refusal_rate": r["benign_refusal_rate"],
                "benign_refusal_ci": f"[{r['benign_refusal_ci_low']:.1f}, {r['benign_refusal_ci_high']:.1f}]",
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary_sorted = pd.concat([
            df_summary.iloc[:2],
            df_summary.iloc[2:].sort_values(by="adv_safety_rate", ascending=False)
        ], ignore_index=True)

        summary_csv_path = os.path.join(results_dir, "summary_comparison.csv")
        df_summary_sorted.to_csv(summary_csv_path, index=False)

        print("\n" + "=" * 80)
        print("🏆 FINAL COMPARISON SUMMARY TABLE (Sorted by Adversarial Safety %)")
        print("=" * 80)
        print(df_summary_sorted.to_string(index=False))
        print("=" * 80 + "\n")

        # Log Final Tables and Best Summary to W&B
        if use_wandb and wandb.run is not None:
            wandb.log({
                "results/sweep_table": wandb.Table(dataframe=df_results),
                "results/summary_table": wandb.Table(dataframe=df_summary_sorted),
            })

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
    parser = argparse.ArgumentParser(description="Run Steering Vector Sweep Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="steering/steering_config.yaml",
        help="Path to steering config YAML",
    )
    args = parser.parse_args()
    run_sweep(config_path=args.config)


if __name__ == "__main__":
    main()
