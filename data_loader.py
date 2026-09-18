import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# WildGuard is the unified replacement; legacy checkers kept as fallback
try:
    from steering.wildguard_eval import WildGuardChecker
except Exception:
    WildGuardChecker = None
try:
    from refusal_checker import RefusalChecker
except Exception:
    RefusalChecker = WildGuardChecker  # type: ignore
try:
    from gliguard_checker import GLiGuardChecker
except Exception:
    GLiGuardChecker = WildGuardChecker  # type: ignore

class AlignmentDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        max_samples=None,
        max_length=256,
        max_memory_length=128,
        constitution_path="constitution.txt",
        dataset_name="PKU-Alignment/PKU-SafeRLHF",
        filter_refusal_only=False,
        refusal_model_name="natong19/refusal_classifier",
        refusal_filter_batch_size=64,
        eval_mode="adversarial",
        filter_eval_with_guard=True,
        guard_model_name="fastino/gliguard-LLMGuardrails-300M",
        guard_filter_batch_size=64,
        seed=42,
        device=None,
    ):
        print(f"Loading {dataset_name} dataset ({split} split)...")
        self.dataset = load_dataset(dataset_name, split=split)

        print("Filtering dataset based on safety conditions...")
        if split == "train":
            # Train: Exactly one is safe, exactly one is unsafe (True != False)
            self.dataset = self.dataset.filter(
                lambda x: x['is_response_0_safe'] != x['is_response_1_safe']
            )

            # Optional refusal filtering: Evaluate safe response with WildGuard (replaces RefusalChecker)
            if filter_refusal_only:
                print("Evaluating and filtering dataset to retain ONLY safe responses that are explicit refusals (WildGuard)...")
                # Prefer WildGuard if available
                if WildGuardChecker is not None and "wildguard" in refusal_model_name.lower():
                    checker = WildGuardChecker(model_name=refusal_model_name, device=device)  # type: ignore
                    filter_is_wildguard = True
                else:
                    checker = RefusalChecker(model_name=refusal_model_name, device=device)  # type: ignore
                    filter_is_wildguard = False

                def filter_refusal_fn(batch):
                    safe_texts = []
                    safer_ids = batch.get("safer_response_id", batch.get("safe_response_id", [0] * len(batch["response_0"])))
                    for safer_id, r0, r1 in zip(safer_ids, batch["response_0"], batch["response_1"]):
                        safe_texts.append(r0 if safer_id == 0 else r1)
                    if filter_is_wildguard:
                        return checker.is_refusal(safe_texts)  # type: ignore
                    return checker.is_refusal(safe_texts)  # type: ignore

                self.dataset = self.dataset.filter(
                    filter_refusal_fn, batched=True, batch_size=refusal_filter_batch_size
                )
                print(f"Refusal filter complete: {len(self.dataset)} refusal samples available.")

                # Free classifier VRAM
                del checker
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Eval: adversarial (both responses unsafe) vs safe/benign (both responses safe)
            print(f"Loading (eval_mode={eval_mode})...")
            if eval_mode == "safe":
                self.dataset = self.dataset.filter(
                    lambda x: x['is_response_0_safe'] and x['is_response_1_safe']
                )
            else:
                self.dataset = self.dataset.filter(
                    lambda x: not x['is_response_0_safe'] and not x['is_response_1_safe']
                )

            # Pre-shuffle and select candidate pool if max_samples is provided to speed up evaluation filtering
            if max_samples is not None:
                if seed is not None:
                    self.dataset = self.dataset.shuffle(seed=seed)
                candidate_limit = min(len(self.dataset), max_samples * 4)
                self.dataset = self.dataset.select(range(candidate_limit))

            # Filter input prompts using WildGuard (replaces GLiGuard) when model is wildguard
            if filter_eval_with_guard:
                use_wildguard = WildGuardChecker is not None and "wildguard" in guard_model_name.lower()
                label = "WildGuard" if use_wildguard else "GLiGuard"
                print(f"Filtering input prompts using {label} ({guard_model_name}) for eval_mode='{eval_mode}'...")
                if use_wildguard:
                    guard = WildGuardChecker(model_name=guard_model_name, device=device)  # type: ignore
                else:
                    guard = GLiGuardChecker(model_name=guard_model_name, device=device)  # type: ignore

                def filter_guard_prompt_fn(batch):
                    prompts = batch["prompt"]
                    if eval_mode == "safe":
                        return guard.is_prompt_safe(prompts, batch_size=guard_filter_batch_size)  # type: ignore
                    else:
                        return guard.is_prompt_unsafe(prompts, batch_size=guard_filter_batch_size)  # type: ignore

                self.dataset = self.dataset.filter(
                    filter_guard_prompt_fn, batched=True, batch_size=guard_filter_batch_size
                )
                print(f"{label} prompt filtering complete: {len(self.dataset)} samples retained.")

                del guard
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if seed is not None and split == "train":
            self.dataset = self.dataset.shuffle(seed=seed)

        # Take a subset if max_samples is specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 🛡️ THE AXIOM BLOCK (The rules we will inject into the middle layers)
        with open(constitution_path, 'r', encoding='utf-8') as file:
            self.constitution = file.read()

        self.memory_ids = self.tokenizer(
            self.constitution, max_length=max_memory_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # PKU dataset has 'safer_response_id' to tell us which response to learn from
        safer_id = item['safer_response_id'] 
        safe_response = item[f'response_{safer_id}']
        user_prompt = item['prompt']
        
        # Format using the model's standard chat template
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # We need to know where the prompt ends so we don't train the model on the user's text!
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.squeeze(0)
        prompt_len = len(prompt_ids)
        
        # Full text: Prompt + Safe Response
        full_text = prompt_text + safe_response + self.tokenizer.eos_token
        encodings = self.tokenizer(
            full_text, max_length=self.max_length, 
            padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "memory_ids": self.memory_ids, # The immutable rules
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0),
            "prompt_len": prompt_len, # We will use this to mask the loss
            "prompt_text": user_prompt  # Added this so eval.py can easily reference the raw prompt
        }