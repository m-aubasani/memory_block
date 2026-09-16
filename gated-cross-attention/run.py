import os
import sys
import argparse
import yaml
import torch
import gc
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Ensure project root and current dir are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [PROJECT_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import root modules and local modules
from data_loader import AlignmentDataset
from eval import run_evaluation
from model import AlignedInjectedLLM
from train import train_model
from inference import InjectedGenerator


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        fallback = os.path.join(CURRENT_DIR, "config.yaml")
        if os.path.exists(fallback):
            config_path = fallback
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    default_config = os.path.join(CURRENT_DIR, "config.yaml") if os.path.exists(os.path.join(CURRENT_DIR, "config.yaml")) else "config.yaml"
    parser = argparse.ArgumentParser(description="Run Alignment Injection Pipeline (Gated Cross-Attention)")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    wandb_cfg = config.get("wandb", {})

    # Set random seed for reproducibility
    seed = config.get("seed", data_cfg.get("seed", eval_cfg.get("seed", 42)))
    set_seed(seed)

    # Initialize Weights & Biases if enabled
    use_wandb = wandb_cfg.get("enabled", True)
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "memory-block-alignment"),
            entity=wandb_cfg.get("entity", "mr_letters-personal"),
            name=wandb_cfg.get("run_name", None),
            mode=wandb_cfg.get("mode", "online"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )

    try:
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
        print(f"Using Seed: {seed}")
        
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

        # Resolve constitution path
        constitution_path = data_cfg.get("constitution_path", "constitution.txt")
        if not os.path.exists(constitution_path):
            fallback_constitution = os.path.join(PROJECT_ROOT, constitution_path)
            if os.path.exists(fallback_constitution):
                constitution_path = fallback_constitution

        # 3. PREPARE DATASET
        print("\n--- Preparing Data ---")
        train_dataset = AlignmentDataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=data_cfg.get("train_max_samples", 500),
            max_length=data_cfg.get("max_seq_length", 256),
            max_memory_length=data_cfg.get("max_memory_length", 128),
            constitution_path=constitution_path,
            dataset_name=data_cfg.get("dataset_name", "PKU-Alignment/PKU-SafeRLHF"),
            filter_refusal_only=data_cfg.get("filter_refusal_only", False),
            refusal_model_name=data_cfg.get(
                "refusal_classifier_model",
                eval_cfg.get("refusal_classifier_model", "natong19/refusal_classifier"),
            ),
            refusal_filter_batch_size=data_cfg.get("refusal_filter_batch_size", 64),
            seed=seed,
            device=device,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 2),
            shuffle=True,
        )
        lr = float(train_cfg.get("learning_rate", 3e-4))
        gate_lr = float(train_cfg.get("gate_learning_rate", lr * 100))

        gate_params = [p for n, p in model.named_parameters() if p.requires_grad and "gate" in n]
        other_params = [p for n, p in model.named_parameters() if p.requires_grad and "gate" not in n]

        optimizer = optim.AdamW([
            {"params": other_params, "lr": lr},
            {"params": gate_params, "lr": gate_lr},
        ])

        # 4. TRAINING PHASE
        # Sanity check 
        print("\n--- Sanity Check of Trainable Parameters ---")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print("\n--- Starting Training Phase ---")
        train_model(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            device=device, 
            epochs=train_cfg.get("epochs", 1),
            log_interval=train_cfg.get("log_interval", 20),
        )

        print("\n--- Clearing VRAM for Evaluation ---")                                                                                   
        del optimizer                                                                                                                     
        del train_loader                                                                                                                  
        del train_dataset                                                                                                                 
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()  

        # 5. EVALUATION PHASE
        print("\n--- Starting Evaluation Phase ---")
        model.eval()
        generator = InjectedGenerator(model)
        
        num_eval_samples = eval_cfg.get("num_samples", 20)
        eval_batch_size = eval_cfg.get("batch_size", 64)
        max_new_tokens = eval_cfg.get("max_new_tokens", 100)
        dataset_name = data_cfg.get("dataset_name", "PKU-Alignment/PKU-SafeRLHF")
        refusal_model_name = eval_cfg.get("refusal_classifier_model", "natong19/refusal_classifier")
        guard_model_name = eval_cfg.get("guard_model", "fastino/gliguard-LLMGuardrails-300M")
        filter_eval_with_guard = data_cfg.get("filter_eval_with_guard", True)

        # 5.1 Adversarial Evaluation (both responses unsafe)
        print("\n[1/2] Running Adversarial Evaluation (both responses unsafe, evaluated with GLiGuard)...")
        adv_output_csv = eval_cfg.get("output_adversarial_csv_path", eval_cfg.get("output_csv_path", "alignment_eval_adversarial_results.csv"))
        run_evaluation(
            model=model, 
            tokenizer=tokenizer, 
            generator=generator, 
            num_samples=num_eval_samples,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
            constitution_path=constitution_path,
            dataset_name=dataset_name,
            refusal_model_name=refusal_model_name,
            guard_model_name=guard_model_name,
            filter_eval_with_guard=filter_eval_with_guard,
            output_path=adv_output_csv,
            eval_mode="adversarial",
            seed=seed,
        )

        # 5.2 Safe/Benign Evaluation (both responses safe)
        print("\n[2/2] Running Safe/Benign Evaluation (both responses safe, evaluated with RefusalChecker)...")
        safe_output_csv = eval_cfg.get("output_safe_csv_path", "alignment_eval_safe_results.csv")
        run_evaluation(
            model=model, 
            tokenizer=tokenizer, 
            generator=generator, 
            num_samples=num_eval_samples,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
            constitution_path=constitution_path,
            dataset_name=dataset_name,
            refusal_model_name=refusal_model_name,
            guard_model_name=guard_model_name,
            filter_eval_with_guard=filter_eval_with_guard,
            output_path=safe_output_csv,
            eval_mode="safe",
            seed=seed,
        )
        
        print(f"\n🎉 Pipeline Complete! Check '{adv_output_csv}' and '{safe_output_csv}' for details.")

    finally:
        if use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    main()
