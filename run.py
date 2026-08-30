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

def main():
    # 1. SETUP DEVICE & TOKENIZER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using Device: {device}")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    # 2. LOAD & WRAP MODEL
    print("Loading Base Model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    
    print("Wrapping model with Axiomatic Injection Blocks...")
    model = AlignedInjectedLLM(
        base_model=base_model, 
        hidden_size=base_model.config.hidden_size,
        extraction_layer=8,
        injection_layers=[8, 16]
    ).to(device)

    # Freeze base model, unfreeze custom blocks
    for param in model.base_model.parameters(): 
        param.requires_grad = False
    for param in model.constraint_encoder.parameters(): 
        param.requires_grad = True
    for param in model.injection_modules.parameters(): 
        param.requires_grad = True

    # 3. PREPARE DATASET
    print("\n--- Preparing Data ---")
    train_dataset = AlignmentDataset(tokenizer=tokenizer, split="train", max_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    # 4. TRAINING PHASE
    print("\n--- Starting Training Phase ---")
    # This calls the training loop from your train.py file
    train_model(
        model=model, 
        dataloader=train_loader, 
        optimizer=optimizer, 
        device=device, 
        epochs=1
    )

    # 5. EVALUATION PHASE
    print("\n--- Starting Evaluation Phase ---")
    model.eval()
    generator = InjectedGenerator(model) # From inference.py
    
    # This calls the eval loop from your eval.py file and prints the dataframe
    run_evaluation(
        model=model, 
        tokenizer=tokenizer, 
        generator=generator, 
        num_samples=20
    )
    
    print("\n🎉 Pipeline Complete! Check 'alignment_eval_results.csv' for details.")

if __name__ == "__main__":
    main()