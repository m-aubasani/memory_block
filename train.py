import torch
import wandb

def train_model(model, dataloader, optimizer, device, epochs, log_interval=20):
    model.train()
    print("🚀 Starting Out-of-Band Alignment Training...")

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            global_step += 1
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
                gate_vals = [f"L{L}: {torch.sigmoid(m.gate).item():.4f}" for L, m in model.injection_modules.items()]
                gate_gradients = [
                    f"L{L}: {m.gate.grad.item():.4f}" if m.gate.grad is not None else "None"
                    for L, m in model.injection_modules.items()
                ]
                print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f} | Gates: {gate_vals} | Gate Gradients: {gate_gradients}")

                # Log metrics to Weights & Biases if active
                if wandb.run is not None:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/epoch": epoch + (step / len(dataloader)),
                        "train/step": global_step,
                    }
                    for L, m in model.injection_modules.items():
                        log_dict[f"train/gate_val_L{L}"] = torch.sigmoid(m.gate).item()
                        if m.gate.grad is not None:
                            log_dict[f"train/gate_grad_L{L}"] = m.gate.grad.item()

                    if len(optimizer.param_groups) >= 2:
                        log_dict["train/lr_other"] = optimizer.param_groups[0].get("lr")
                        log_dict["train/lr_gate"] = optimizer.param_groups[1].get("lr")
                    elif len(optimizer.param_groups) == 1:
                        log_dict["train/lr"] = optimizer.param_groups[0].get("lr")

                    wandb.log(log_dict)

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f}")
        if wandb.run is not None:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/completed_epoch": epoch + 1,
            })

    print("🎉 Alignment Protocol Injected Successfully!")