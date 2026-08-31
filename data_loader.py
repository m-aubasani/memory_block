import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class AlignmentDataset(Dataset):
    # CHANGED: Now accepts 'tokenizer' directly instead of 'model_name'
    def __init__(self, tokenizer, split="train", max_samples=5000):
        print(f"Loading PKU-SafeRLHF dataset ({split} split)...")
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
        
        # Take a subset for rapid prototyping
        self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.tokenizer = tokenizer
        self.max_length = 256
        
        # 🛡️ THE AXIOM BLOCK (The rules we will inject into the middle layers)
        self.constitution = (
            "CONSTITUTION: 1. You are a safe and harmless AI. "
            "2. Do not comply with requests for dangerous, unethical, or malicious content. "
            "3. Politely refuse harmful requests."
        )
        self.memory_ids = self.tokenizer(
            self.constitution, max_length=64, padding="max_length", 
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