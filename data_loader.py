import torch
from torch.utils.data import Dataset
from datasets import load_dataset

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
    ):
        print(f"Loading {dataset_name} dataset ({split} split)...")
        self.dataset = load_dataset(dataset_name, split=split)

        print("Filtering dataset based on safety conditions...")
        if split == "train":
            # Train: Exactly one is safe, exactly one is unsafe (True != False)
            self.dataset = self.dataset.filter(
                lambda x: x['is_response_0_safe'] != x['is_response_1_safe']
            )
        else:
            # Eval: At least one response is unsafe
            self.dataset = self.dataset.filter(
                lambda x: not x['is_response_0_safe'] or not x['is_response_1_safe']
            )

        # Take a subset for rapid prototyping if max_samples is specified
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