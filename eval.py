import torch
import pandas as pd
import wandb
from tqdm import tqdm
from data_loader import AlignmentDataset
from refusal_checker import RefusalChecker
from gliguard_checker import GLiGuardChecker


def run_evaluation(
    model,
    tokenizer,
    generator,
    num_samples=100,
    max_new_tokens=100,
    constitution_path="constitution.txt",
    dataset_name="PKU-Alignment/PKU-SafeRLHF",
    refusal_model_name="natong19/refusal_classifier",
    guard_model_name="fastino/gliguard-LLMGuardrails-300M",
    filter_eval_with_guard=True,
    output_path="alignment_eval_results.csv",
    eval_mode="adversarial",
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For adversarial set (unsafe prompts): use GLiGuard to evaluate response safety (no RefusalChecker)
    # For safe set (safe prompts): use RefusalChecker to evaluate refusal / over-refusal rate
    checker = None
    guard = None

    if eval_mode == "safe":
        checker = RefusalChecker(model_name=refusal_model_name, device=device)
    else:
        guard = GLiGuardChecker(model_name=guard_model_name, device=device)

    alignment_dataset = AlignmentDataset(
        tokenizer=tokenizer,
        split="test",
        max_samples=num_samples,
        constitution_path=constitution_path,
        dataset_name=dataset_name,
        eval_mode=eval_mode,
        filter_eval_with_guard=filter_eval_with_guard,
        guard_model_name=guard_model_name,
        seed=seed,
        device=device,
    )

    dataset = alignment_dataset.dataset

    # 🛡️ THE AXIOM BLOCK (The rules we will inject into the middle layers)
    with open(constitution_path, 'r', encoding='utf-8') as file:
        constitution = file.read()
    memory_ids = tokenizer(constitution, return_tensors="pt").input_ids.to(device)
    
    results = []
    
    mode_label = "Adversarial" if eval_mode != "safe" else "Safe/Benign"
    metric_label = "Safety Rate %" if eval_mode != "safe" else "Refusal Rate %"
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

        item_result = {
            "Prompt": user_prompt,
            "Base_Response": res_base,
            "SysPrompt_Response": res_sys,
            "Injected_Response": res_inj,
        }

        if eval_mode != "safe":
            # Unsafe / Adversarial Eval Set: Evaluate response safety with GLiGuard
            item_result["Base_Safe"] = guard.is_response_safe(res_base, prompts=user_prompt)
            item_result["SysPrompt_Safe"] = guard.is_response_safe(res_sys, prompts=user_prompt)
            item_result["Injected_Safe"] = guard.is_response_safe(res_inj, prompts=user_prompt)
        else:
            # Safe / Benign Eval Set: Evaluate refusal rate with RefusalChecker
            item_result["Base_Refused"] = checker.is_refusal(res_base)
            item_result["SysPrompt_Refused"] = checker.is_refusal(res_sys)
            item_result["Injected_Refused"] = checker.is_refusal(res_inj)

        results.append(item_result)

    # Clean up guard/checker from memory
    if guard is not None:
        del guard
    if checker is not None:
        del checker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate and print metrics
    df = pd.DataFrame(results)

    if eval_mode != "safe":
        base_score = df["Base_Safe"].mean() * 100
        sys_score = df["SysPrompt_Safe"].mean() * 100
        inj_score = df["Injected_Safe"].mean() * 100
        metric_key = "safety_rate"
    else:
        base_score = df["Base_Refused"].mean() * 100
        sys_score = df["SysPrompt_Refused"].mean() * 100
        inj_score = df["Injected_Refused"].mean() * 100
        metric_key = "refusal_rate"

    print("\n" + "=" * 55)
    print(f"🎯 {mode_label.upper()} EVALUATION RESULTS ({metric_label})")
    print("=" * 55)
    print(f"1. Baseline (No Protection): {base_score:.1f}%")
    print(f"2. Traditional (System Prompt): {sys_score:.1f}%")
    print(f"3. Axiomatic Injection (Ours): {inj_score:.1f}%")
    print("=" * 55)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved detailed outputs to '{output_path}'")

    # Log evaluation results to Weights & Biases if active
    if wandb.run is not None:
        prefix = f"eval_{eval_mode}"
        eval_metrics = {
            f"{prefix}/baseline_{metric_key}": base_score,
            f"{prefix}/sysprompt_{metric_key}": sys_score,
            f"{prefix}/injected_{metric_key}": inj_score,
            f"{prefix}/num_samples": len(df),
            f"{prefix}/results_table": wandb.Table(dataframe=df),
        }
        wandb.log(eval_metrics)

        # Update summary for easy dashboard sorting/filtering
        wandb.run.summary[f"{prefix}_baseline_{metric_key}"] = base_score
        wandb.run.summary[f"{prefix}_sysprompt_{metric_key}"] = sys_score
        wandb.run.summary[f"{prefix}_injected_{metric_key}"] = inj_score

    return df