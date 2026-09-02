import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your custom modules
from data_loader import AlignmentDataset
from model import AlignedInjectedLLM
from train import train_model
from inference import InjectedGenerator
from eval import run_evaluation


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Alignment Injection Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    # Map dtype string to torch dtype
    dtype_str = model_cfg.get("dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # 1. SETUP DEVICE & TOKENIZER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using Device: {device}")
    
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    # 2. LOAD & WRAP MODEL
    print(f"Loading Base Model ({model_name}) in {dtype_str}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype
    ).to(device)
    
    print("Wrapping model with Axiomatic Injection Blocks...")
    model = AlignedInjectedLLM(
        base_model=base_model, 
        hidden_size=base_model.config.hidden_size,
        layer_pairs=model_cfg.get("layer_pairs", None),
        extraction_layers=model_cfg.get("extraction_layers", [8, 16]),
        injection_layers=model_cfg.get("injection_layers", [8, 16]),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 2),
        num_heads=model_cfg.get("num_heads", 8),
    ).to(device, dtype=torch_dtype)

    # Freeze base model, unfreeze custom blocks
    for param in model.base_model.parameters(): 
        param.requires_grad = False
    for param in model.constraint_encoders.parameters(): 
        param.requires_grad = True
    for param in model.injection_modules.parameters(): 
        param.requires_grad = True

    # 3. PREPARE DATASET
    print("\n--- Preparing Data ---")
    train_dataset = AlignmentDataset(
        tokenizer=tokenizer,
        split="train",
        max_samples=data_cfg.get("train_max_samples", 500),
        max_length=data_cfg.get("max_seq_length", 256),
        max_memory_length=data_cfg.get("max_memory_length", 128),
        constitution_path=data_cfg.get("constitution_path", "constitution.txt"),
        dataset_name=data_cfg.get("dataset_name", "PKU-Alignment/PKU-SafeRLHF"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=True,
    )
    lr = float(train_cfg.get("learning_rate", 3e-4))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 4. TRAINING PHASE
    print("\n--- Starting Training Phase ---")
    train_model(
        model=model, 
        dataloader=train_loader, 
        optimizer=optimizer, 
        device=device, 
        epochs=train_cfg.get("epochs", 1),
        log_interval=train_cfg.get("log_interval", 20),
    )

    # 5. EVALUATION PHASE
    print("\n--- Starting Evaluation Phase ---")
    model.eval()
    generator = InjectedGenerator(model)
    
    output_csv = eval_cfg.get("output_csv_path", "alignment_eval_results.csv")
    run_evaluation(
        model=model, 
        tokenizer=tokenizer, 
        generator=generator, 
        num_samples=eval_cfg.get("num_samples", 20),
        max_new_tokens=eval_cfg.get("max_new_tokens", 100),
        constitution_path=data_cfg.get("constitution_path", "constitution.txt"),
        dataset_name=data_cfg.get("dataset_name", "PKU-Alignment/PKU-SafeRLHF"),
        refusal_model_name=eval_cfg.get("refusal_classifier_model", "natong19/refusal_classifier"),
        output_path=output_csv,
    )
    
    print(f"\n🎉 Pipeline Complete! Check '{output_csv}' for details.")

if __name__ == "__main__":
    main()