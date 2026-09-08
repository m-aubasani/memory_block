import torch
import pandas as pd
import wandb
from datasets import load_dataset
from tqdm import tqdm
# from inference import InjectedGenerator
# from model import AlignedInjectedLLM
from refusal_checker import RefusalChecker


def run_evaluation(
    model,
    tokenizer,
    generator,
    num_samples=100,
    max_new_tokens=100,
    constitution_path="constitution.txt",
    dataset_name="PKU-Alignment/PKU-SafeRLHF",
    refusal_model_name="natong19/refusal_classifier",
    output_path="alignment_eval_results.csv",
    eval_mode="adversarial",
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checker = RefusalChecker(model_name=refusal_model_name, device=device)

    print(f"Loading {dataset_name} test dataset (eval_mode={eval_mode})...")
    # Use the test split to ensure it's data the model hasn't seen
    dataset = load_dataset(dataset_name, split="test")

    # Filter based on safety conditions
    if eval_mode == "safe":
        # Safe/Benign: Both responses safe
        dataset = dataset.filter(
            lambda x: x['is_response_0_safe'] and x['is_response_1_safe']
        )
    else:
        # Adversarial: Both responses unsafe
        dataset = dataset.filter(
            lambda x: not x['is_response_0_safe'] and not x['is_response_1_safe']
        )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # 🛡️ THE AXIOM BLOCK (The rules we will inject into the middle layers)
    with open(constitution_path, 'r', encoding='utf-8') as file:
        constitution = file.read()
    memory_ids = tokenizer(constitution, return_tensors="pt").input_ids.to(device)
    
    results = []
    
    mode_label = "Adversarial" if eval_mode != "safe" else "Safe/Benign"
    print(f"Evaluating {len(dataset)} {mode_label.lower()} prompts...")
    for item in tqdm(dataset):
        user_prompt = item['prompt']
        
        # ==========================================
        # SETUP 1: BASELINE (No System Prompt)
        # ==========================================
        msg_base = [{"role": "user", "content": user_prompt}]
        ids_base = tokenizer(tokenizer.apply_chat_template(msg_base, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to(device)
        
        # ==========================================
        # SETUP 2: SYSTEM PROMPT (Traditional Alignment)
        # ==========================================
        msg_sys = [
            {"role": "system", "content": constitution},
            {"role": "user", "content": user_prompt}
        ]
        ids_sys = tokenizer(tokenizer.apply_chat_template(msg_sys, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to(device)
        
        # ==========================================
        # GENERATION (using torch.no_grad() for memory efficiency)
        # ==========================================
        with torch.no_grad():
            # 1. Base Generation (No hooks active!)
            out_base = model.base_model.generate(ids_base, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            res_base = tokenizer.decode(out_base[0][ids_base.shape[1]:], skip_special_tokens=True)
            
            # 2. System Prompt Generation (No hooks active!)
            out_sys = model.base_model.generate(ids_sys, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            res_sys = tokenizer.decode(out_sys[0][ids_sys.shape[1]:], skip_special_tokens=True)
            
            # 3. Injected Generation (Hooks Active, NO system prompt)
            out_inj = generator.generate(ids_base, memory_ids=memory_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            res_inj = tokenizer.decode(out_inj[0][ids_base.shape[1]:], skip_special_tokens=True)
            
        results.append({
            "Prompt": user_prompt,
            "Base_Response": res_base,
            "SysPrompt_Response": res_sys,
            "Injected_Response": res_inj,
            "Base_Refused": checker.is_refusal(res_base),
            "SysPrompt_Refused": checker.is_refusal(res_sys),
            "Injected_Refused": checker.is_refusal(res_inj)
        })
        
    # Calculate and print metrics
    df = pd.DataFrame(results)
    
    base_score = df["Base_Refused"].mean() * 100
    sys_score = df["SysPrompt_Refused"].mean() * 100
    inj_score = df["Injected_Refused"].mean() * 100
    
    print("\n" + "="*45)
    print(f"🎯 {mode_label.upper()} EVALUATION RESULTS (Refusal Rate %)")
    print("="*45)
    print(f"1. Baseline (No Protection): {base_score:.1f}%")
    print(f"2. Traditional (System Prompt): {sys_score:.1f}%")
    print(f"3. Axiomatic Injection (Ours): {inj_score:.1f}%")
    print("="*45)
    
    # Save to CSV for manual review or passing to an LLM-Judge later
    df.to_csv(output_path, index=False)
    print(f"Saved detailed outputs to '{output_path}'")

    # Log evaluation results to Weights & Biases if active
    if wandb.run is not None:
        prefix = f"eval_{eval_mode}"
        eval_metrics = {
            f"{prefix}/baseline_refusal_rate": base_score,
            f"{prefix}/sysprompt_refusal_rate": sys_score,
            f"{prefix}/injected_refusal_rate": inj_score,
            f"{prefix}/delta_inj_vs_base": inj_score - base_score,
            f"{prefix}/delta_inj_vs_sys": inj_score - sys_score,
            f"{prefix}/num_samples": len(df),
            f"{prefix}/results_table": wandb.Table(dataframe=df),
        }
        wandb.log(eval_metrics)

        # Update summary for easy dashboard sorting/filtering
        wandb.run.summary[f"{prefix}_baseline_refusal_rate"] = base_score
        wandb.run.summary[f"{prefix}_sysprompt_refusal_rate"] = sys_score
        wandb.run.summary[f"{prefix}_injected_refusal_rate"] = inj_score

    return df
    

# Run it
# run_evaluation(model, tokenizer, generator, num_samples=50)