import torch

def train_model(model, dataloader, optimizer, device, epochs, log_interval=20):
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
            
            if step % log_interval == 0:
                # Monitor the gates! Watch them rise from 0.0 as the network learns to rely on the Constitution
                gate_vals = [f"L{L}: {torch.tanh(m.gate).item():.4f}" for L, m in model.injection_modules.items()]
                print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f} | Gates: {gate_vals}")

        print(f"✅ Epoch {epoch} Complete | Avg Loss: {epoch_loss / len(dataloader):.4f}")

    print("🎉 Alignment Protocol Injected Successfully!")