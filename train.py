import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from model import AlignedInjectedLLM
from transformers import AutoModelForCausalLM
from data_loader import AlignmentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # 1. Initialize Base Model
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# print("Loading base LLM in bfloat16...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.bfloat16
# ).to(device)

# # 2. Wrap model in our Axiom Architecture
# model = AlignedInjectedLLM(
#     base_model=base_model, 
#     hidden_size=base_model.config.hidden_size, 
#     vocab_size=base_model.config.vocab_size, 
#     injection_layers=[8, 16] # Inject rules at layers 8 and 16
# ).to(device)

# # --- PEFT: FREEZE BASE MODEL ---
# for param in model.base_model.parameters():
#     param.requires_grad = False

# # Ensure injected pathways are trainable
# for param in model.constraint_encoder.parameters():
#     param.requires_grad = True
# for param in model.injection_modules.parameters():
#     param.requires_grad = True

# print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 3. Dataloader & Optimizer
# dataset = AlignmentDataset(model_name=model_name, max_samples=2000)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # Batch size 4 for Colab T4
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

# 4. TRAINING LOOP
def train_model(model, dataloader, optimizer, device, epochs=1):
    model.train()
    print("🚀 Starting Out-of-Band Alignment Training...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            memory_ids = batch["memory_ids"].to(device)
            prompt_lens = batch["prompt_len"]
            
            # CRITICAL: Loss Masking
            labels = input_ids.clone()
            # 1. Ignore padding tokens
            labels[attention_mask == 0] = -100 
            
            # 2. Ignore the user's adversarial prompt (we don't want to train the AI to generate jailbreaks!)
            for i, p_len in enumerate(prompt_lens):
                # Mask out everything up to the length of the prompt
                labels[i, :p_len] = -100
                
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_ids=memory_ids,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if step % 20 == 0:
                # Monitor the gates! Watch them rise from 0.0 as the network learns to rely on the Constitution
                gate_vals = [f"L{L}: {torch.tanh(m.gate).item():.4f}" for L, m in model.injection_modules.items()]
                print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f} | Gates: {gate_vals}")

        print(f"✅ Epoch {epoch} Complete | Avg Loss: {epoch_loss / len(dataloader):.4f}")

    print("🎉 Alignment Protocol Injected Successfully!")