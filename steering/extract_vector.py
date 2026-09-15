import os
import sys
import json
import time
import argparse
import yaml
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_yaml_config(config_path: str):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def extract_steering_vectors(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dtype_str: str = "bfloat16",
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    layers: list = None,
    n_pairs: int = 150,
    seed: int = 42,
    output_dir: str = "steering/vectors",
    batch_size: int = 4,
    device: str = None,
):
    if layers is None:
        layers = [7, 10, 14, 18, 21]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"\n=======================================================")
    print(f"🚀 EXTRACTING STEERING VECTORS (CAA / Difference-in-Means)")
    print(f"=======================================================")
    print(f"Model:        {model_name} ({dtype_str})")
    print(f"Dataset:      {dataset_name} (train split)")
    print(f"Target layers:{layers}")
    print(f"N pairs:      {n_pairs}")
    print(f"Seed:         {seed}")
    print(f"Device:       {device}")
    print(f"Output dir:   {output_dir}")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Load and filter dataset pairs
    print(f"Loading '{dataset_name}' train split...")
    raw_dataset = load_dataset(dataset_name, split="train")

    # Filter where exactly one response is safe
    contrastive_dataset = raw_dataset.filter(
        lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]
    )

    if seed is not None:
        contrastive_dataset = contrastive_dataset.shuffle(seed=seed)

    selected_dataset = contrastive_dataset.select(range(min(n_pairs, len(contrastive_dataset))))
    actual_pairs = len(selected_dataset)
    print(f"Selected {actual_pairs} contrastive pairs for vector extraction.")

    # 3. Format full chat conversations for safe and unsafe responses
    safe_texts = []
    unsafe_texts = []

    for item in selected_dataset:
        prompt = item["prompt"]
        if item["is_response_0_safe"]:
            safe_resp = item["response_0"]
            unsafe_resp = item["response_1"]
        else:
            safe_resp = item["response_1"]
            unsafe_resp = item["response_0"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        safe_full = prompt_text + safe_resp + (tokenizer.eos_token or "")
        unsafe_full = prompt_text + unsafe_resp + (tokenizer.eos_token or "")

        safe_texts.append(safe_full)
        unsafe_texts.append(unsafe_full)

    # 4. Load Base Model
    print(f"\nLoading base model '{model_name}'...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
    ).to(device)
    base_model.eval()

    # 5. Extract activations per layer
    safe_activations = {l: [] for l in layers}
    unsafe_activations = {l: [] for l in layers}

    def collect_activations(texts, act_dict, desc):
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = base_model(
                    input_ids=encodings.input_ids,
                    attention_mask=encodings.attention_mask,
                    output_hidden_states=True,
                )

            # Find last non-padded token position for each sequence in the batch
            # For left-padded inputs, the last non-padded token is at index -1
            # For general safety, compute last token position via attention_mask
            for b_idx in range(len(batch)):
                # In left-padded sequences, valid tokens end at index len - 1
                last_pos = encodings.input_ids.shape[1] - 1

                for layer_idx in layers:
                    # In HuggingFace, hidden_states[0] is embedding output,
                    # hidden_states[layer_idx + 1] is output of layer_idx
                    h_state = outputs.hidden_states[layer_idx + 1][b_idx, last_pos, :].detach().float().cpu()
                    act_dict[layer_idx].append(h_state)

    print("\nCollecting safe response activations...")
    collect_activations(safe_texts, safe_activations, "Safe Activations")

    print("\nCollecting unsafe response activations...")
    collect_activations(unsafe_texts, unsafe_activations, "Unsafe Activations")

    # 6. Compute CAA vectors (mean_safe - mean_unsafe) and save
    print("\nComputing difference-in-means steering vectors...")
    saved_files = []
    for layer_idx in layers:
        safe_tensor = torch.stack(safe_activations[layer_idx])  # [N, hidden_dim]
        unsafe_tensor = torch.stack(unsafe_activations[layer_idx])  # [N, hidden_dim]

        mean_safe = safe_tensor.mean(dim=0)
        mean_unsafe = unsafe_tensor.mean(dim=0)

        # Vector points toward safety
        vector = mean_safe - mean_unsafe
        norm = float(torch.norm(vector, p=2).item())

        vector_data = {
            "vector": vector,
            "layer": layer_idx,
            "norm": norm,
            "n_pairs": actual_pairs,
        }

        output_path = os.path.join(output_dir, f"layer_{layer_idx}.pt")
        torch.save(vector_data, output_path)
        saved_files.append(output_path)
        print(f"  ✓ Layer {layer_idx:2d}: norm = {norm:.4f} -> saved to '{output_path}'")

    # 7. Save metadata
    metadata = {
        "model_name": model_name,
        "dtype": dtype_str,
        "dataset_name": dataset_name,
        "layers": layers,
        "n_pairs": actual_pairs,
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved extraction metadata to '{metadata_path}'")
    print("✅ Vector extraction complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Extract CAA Steering Vectors from Contrastive Safety Data")
    parser.add_argument("--config", type=str, default="steering/steering_config.yaml", help="Path to steering config YAML")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Target layer indices (e.g. --layers 7 10 14 18 21)")
    parser.add_argument("--n-pairs", type=int, default=None, help="Number of contrastive pairs to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="steering/vectors", help="Output directory for extracted vectors")
    parser.add_argument("--model-name", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name")
    parser.add_argument("--constitution-path", type=str, default="constitution.txt", help="Constitution path (accepted for interface consistency)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for hidden state extraction")

    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg.get("model", {})
    extract_cfg = cfg.get("extraction", {})

    layers = args.layers if args.layers is not None else extract_cfg.get("layers", [7, 10, 14, 18, 21])
    n_pairs = args.n_pairs if args.n_pairs is not None else extract_cfg.get("n_pairs", 150)
    seed = args.seed if args.seed is not None else extract_cfg.get("seed", 42)
    model_name = args.model_name if args.model_name is not None else model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct")
    dtype_str = model_cfg.get("dtype", "bfloat16")
    dataset_name = args.dataset_name if args.dataset_name is not None else extract_cfg.get("dataset_name", "PKU-Alignment/PKU-SafeRLHF")
    output_dir = args.output_dir

    extract_steering_vectors(
        model_name=model_name,
        dtype_str=dtype_str,
        dataset_name=dataset_name,
        layers=layers,
        n_pairs=n_pairs,
        seed=seed,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
