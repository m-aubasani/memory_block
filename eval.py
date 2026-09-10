import torch
import pandas as pd
import wandb
from tqdm import tqdm
from data_loader import AlignmentDataset
from refusal_checker import RefusalChecker
from gliguard_checker import GLiGuardChecker


def generate_in_batches(
    generate_fn,
    formatted_prompts,
    tokenizer,
    device,
    batch_size=8,
    max_new_tokens=100,
    desc="Generating",
):
    """
    Batched text generation for causal language models using left-padding.
    """
    responses = []
    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_texts = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = generate_fn(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        for j in range(len(batch_texts)):
            gen_tokens = outputs[j][input_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            responses.append(decoded)

    return responses


def run_evaluation(
    model,
    tokenizer,
    generator,
    num_samples=100,
    batch_size=8,
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

    # 1. Load Evaluation Dataset
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
    prompts = [item['prompt'] for item in dataset]

    # Load Constitution
    with open(constitution_path, 'r', encoding='utf-8') as file:
        constitution = file.read()
    memory_ids = tokenizer(constitution, return_tensors="pt").input_ids.to(device)

    # Configure tokenizer for batched autoregressive generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mode_label = "Adversarial" if eval_mode != "safe" else "Safe/Benign"
    metric_label = "Safety Rate %" if eval_mode != "safe" else "Refusal Rate %"
    print(f"\nEvaluating {len(prompts)} {mode_label.lower()} prompts in batches (batch_size={batch_size})...")

    # Format Prompts for Baseline and Injected
    base_formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # Format Prompts for System Prompt
    sys_formatted_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": constitution},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # ==========================================
    # VARIATION 1: BASELINE BATCH GENERATION
    # ==========================================
    print("\n[Variation 1/3] Generating Baseline responses in batches...")
    base_responses = generate_in_batches(
        generate_fn=model.base_model.generate,
        formatted_prompts=base_formatted_prompts,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="Baseline Generation",
    )

    # ==========================================
    # VARIATION 2: SYSTEM PROMPT BATCH GENERATION
    # ==========================================
    print("\n[Variation 2/3] Generating System Prompt responses in batches...")
    sys_responses = generate_in_batches(
        generate_fn=model.base_model.generate,
        formatted_prompts=sys_formatted_prompts,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="System Prompt Generation",
    )

    # ==========================================
    # VARIATION 3: INJECTED AXIOMATIC BATCH GENERATION
    # ==========================================
    print("\n[Variation 3/3] Generating Injected Axiomatic responses in batches...")
    with generator.injected_context(memory_ids):
        inj_responses = generate_in_batches(
            generate_fn=model.base_model.generate,
            formatted_prompts=base_formatted_prompts,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            desc="Injected Generation",
        )

    # Restore tokenizer padding side
    tokenizer.padding_side = orig_padding_side

    # ==========================================
    # BATCH EVALUATION & SAFETY SCORING
    # ==========================================
    results = []

    if eval_mode != "safe":
        # Unsafe / Adversarial Eval Set: Evaluate response safety with GLiGuard in batches
        print("\nEvaluating responses with GLiGuard in batches...")
        guard = GLiGuardChecker(model_name=guard_model_name, device=device)

        base_safe = guard.is_response_safe(base_responses, prompts=prompts, batch_size=batch_size)
        sys_safe = guard.is_response_safe(sys_responses, prompts=prompts, batch_size=batch_size)
        inj_safe = guard.is_response_safe(inj_responses, prompts=prompts, batch_size=batch_size)

        del guard
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for idx, p in enumerate(prompts):
            results.append({
                "Prompt": p,
                "Base_Response": base_responses[idx],
                "SysPrompt_Response": sys_responses[idx],
                "Injected_Response": inj_responses[idx],
                "Base_Safe": base_safe[idx],
                "SysPrompt_Safe": sys_safe[idx],
                "Injected_Safe": inj_safe[idx],
            })
    else:
        # Safe / Benign Eval Set: Evaluate refusal rate with RefusalChecker in batches
        print("\nEvaluating responses with RefusalChecker in batches...")
        checker = RefusalChecker(model_name=refusal_model_name, device=device)

        base_refused = checker.is_refusal(base_responses)
        sys_refused = checker.is_refusal(sys_responses)
        inj_refused = checker.is_refusal(inj_responses)

        del checker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for idx, p in enumerate(prompts):
            results.append({
                "Prompt": p,
                "Base_Response": base_responses[idx],
                "SysPrompt_Response": sys_responses[idx],
                "Injected_Response": inj_responses[idx],
                "Base_Refused": base_refused[idx],
                "SysPrompt_Refused": sys_refused[idx],
                "Injected_Refused": inj_refused[idx],
            })

    # ==========================================
    # METRICS & COMPARISON REPORT
    # ==========================================
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
    print(f"1. Baseline (No Protection):    {base_score:.1f}%")
    print(f"2. Traditional (System Prompt): {sys_score:.1f}%")
    print(f"3. Axiomatic Injection (Ours):  {inj_score:.1f}%")
    print(f"   ↳ Delta vs Baseline:         {inj_score - base_score:+.1f}%")
    print(f"   ↳ Delta vs System Prompt:    {inj_score - sys_score:+.1f}%")
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

        wandb.run.summary[f"{prefix}_baseline_{metric_key}"] = base_score
        wandb.run.summary[f"{prefix}_sysprompt_{metric_key}"] = sys_score
        wandb.run.summary[f"{prefix}_injected_{metric_key}"] = inj_score

    return df