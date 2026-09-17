import os
import sys
import json
import time
from typing import List, Tuple, Optional
import pandas as pd
from datasets import load_dataset

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_adversarial_prompts(
    dataset_name: str = "jailbreakbench",
    harmbench_url: str = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    num_samples: Optional[int] = None,
    seed: int = 42,
    tokenizer=None,
    filter_eval_with_guard: bool = False,
    guard_model_name: str = "fastino/gliguard-LLMGuardrails-300M",
    device: str = "cuda",
) -> List[str]:
    """
    Loads adversarial prompts from JailbreakBench, HarmBench, or PKU-SafeRLHF.
    """
    dataset_lower = dataset_name.lower()

    if "jailbreak" in dataset_lower or "jbb" in dataset_lower:
        print(f"[DATA] Loading JailbreakBench harmful behaviors ('JailbreakBench/JBB-Behaviors')...")
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        prompts = [item["Goal"] for item in ds]

    elif "harmbench" in dataset_lower:
        print(f"[DATA] Loading HarmBench behaviors from '{harmbench_url}'...")
        df = pd.read_csv(harmbench_url)
        # Standardize prompt column
        if "Behavior" in df.columns:
            prompts = df["Behavior"].dropna().tolist()
        elif "prompt" in df.columns:
            prompts = df["prompt"].dropna().tolist()
        else:
            prompts = df.iloc[:, 0].dropna().tolist()

    elif "pku" in dataset_lower:
        print("[DATA] Loading PKU-SafeRLHF adversarial evaluation split...")
        from data_loader import AlignmentDataset
        adv_dataset = AlignmentDataset(
            tokenizer=tokenizer,
            split="test",
            max_samples=num_samples or 300,
            dataset_name="PKU-Alignment/PKU-SafeRLHF",
            eval_mode="adversarial",
            filter_eval_with_guard=filter_eval_with_guard,
            guard_model_name=guard_model_name,
            seed=seed,
            device=device,
        )
        prompts = [item["prompt"] for item in adv_dataset.dataset]

    else:
        # Fallback to load_dataset generic
        print(f"[DATA] Attempting to load custom adversarial dataset '{dataset_name}'...")
        ds = load_dataset(dataset_name, split="test" if "test" in dataset_name else "train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        prompts = [item[col] for item in ds]

    if seed is not None and "pku" not in dataset_lower:
        rng = pd.Series(prompts).sample(frac=1.0, random_state=seed).tolist()
        prompts = rng

    if num_samples is not None and num_samples < len(prompts):
        prompts = prompts[:num_samples]

    print(f"[DATA] Loaded {len(prompts)} adversarial prompts ({dataset_name}).")
    return prompts


def load_benign_prompts(
    dataset_name: str = "xstest",
    num_samples: Optional[int] = None,
    seed: int = 42,
    tokenizer=None,
    filter_eval_with_guard: bool = False,
    guard_model_name: str = "fastino/gliguard-LLMGuardrails-300M",
    device: str = "cuda",
) -> List[str]:
    """
    Loads benign/safe prompts from XSTest, JailbreakBench (benign split), or PKU-SafeRLHF.
    """
    dataset_lower = dataset_name.lower()

    if "xstest" in dataset_lower:
        print("[DATA] Loading XSTest evaluation prompts ('walledai/XSTest')...")
        try:
            ds = load_dataset("walledai/XSTest", split="test")
            # Filter for safe prompts (over-refusal test set)
            safe_ds = [item["prompt"] for item in ds if item.get("label") == "safe" or "label" not in item]
            prompts = safe_ds if len(safe_ds) > 0 else [item["prompt"] for item in ds]
        except Exception:
            print("[DATA] Fallback: loading XSTest via 'natolambert/xstest-v2-copy'...")
            ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
            prompts = [item["prompt"] for item in ds]

    elif "jailbreak" in dataset_lower or "jbb" in dataset_lower:
        print("[DATA] Loading JailbreakBench benign behaviors ('JailbreakBench/JBB-Behaviors')...")
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
        prompts = [item["Goal"] for item in ds]

    elif "pku" in dataset_lower:
        print("[DATA] Loading PKU-SafeRLHF safe/benign evaluation split...")
        from data_loader import AlignmentDataset
        safe_dataset = AlignmentDataset(
            tokenizer=tokenizer,
            split="test",
            max_samples=num_samples or 300,
            dataset_name="PKU-Alignment/PKU-SafeRLHF",
            eval_mode="safe",
            filter_eval_with_guard=filter_eval_with_guard,
            guard_model_name=guard_model_name,
            seed=seed,
            device=device,
        )
        prompts = [item["prompt"] for item in safe_dataset.dataset]

    else:
        print(f"[DATA] Attempting to load custom benign dataset '{dataset_name}'...")
        ds = load_dataset(dataset_name, split="test" if "test" in dataset_name else "train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        prompts = [item[col] for item in ds]

    if seed is not None and "pku" not in dataset_lower:
        rng = pd.Series(prompts).sample(frac=1.0, random_state=seed).tolist()
        prompts = rng

    if num_samples is not None and num_samples < len(prompts):
        prompts = prompts[:num_samples]

    print(f"[DATA] Loaded {len(prompts)} benign prompts ({dataset_name}).")
    return prompts


def get_or_create_evaluation_suite(
    cache_dir: str = "steering/eval_cache",
    adv_dataset_name: str = "jailbreakbench",
    benign_dataset_name: str = "xstest",
    harmbench_url: str = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    num_samples: Optional[int] = None,
    seed: int = 42,
    tokenizer=None,
    filter_eval_with_guard: bool = False,
    guard_model_name: str = "fastino/gliguard-LLMGuardrails-300M",
    device: str = "cuda",
) -> Tuple[List[str], List[str]]:
    """
    Obtains cached or freshly loaded evaluation splits.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"eval_set_{adv_dataset_name}_{benign_dataset_name}_{num_samples or 'all'}_s{seed}.json"
    )

    if os.path.exists(cache_file):
        print(f"[CACHE] Loading cached evaluation prompts from '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[CACHE] Loaded {len(data['adversarial'])} adversarial and {len(data['safe'])} benign prompts from cache.")
            return data["adversarial"], data["safe"]

    print(f"[EVAL] Building evaluation suite: Adv='{adv_dataset_name}', Benign='{benign_dataset_name}'...")
    adv_prompts = load_adversarial_prompts(
        dataset_name=adv_dataset_name,
        harmbench_url=harmbench_url,
        num_samples=num_samples,
        seed=seed,
        tokenizer=tokenizer,
        filter_eval_with_guard=filter_eval_with_guard,
        guard_model_name=guard_model_name,
        device=device,
    )

    benign_prompts = load_benign_prompts(
        dataset_name=benign_dataset_name,
        num_samples=num_samples,
        seed=seed,
        tokenizer=tokenizer,
        filter_eval_with_guard=filter_eval_with_guard,
        guard_model_name=guard_model_name,
        device=device,
    )

    cache_data = {
        "adversarial_dataset": adv_dataset_name,
        "benign_dataset": benign_dataset_name,
        "adversarial": adv_prompts,
        "safe": benign_prompts,
        "metadata": {
            "num_samples": num_samples,
            "seed": seed,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)
    print(f"[OK] Cached evaluation prompts to '{cache_file}'.\n")

    return adv_prompts, benign_prompts
