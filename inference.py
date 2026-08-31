import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Assuming your classes are saved in model.py
from model import AlignedInjectedLLM 

class InjectedGenerator:
    """
    A wrapper to handle HuggingFace's .generate() method.
    It temporarily attaches the hooks so the memory states are available
    at every step of the autoregressive generation loop.
    """
    def __init__(self, injected_model):
        self.model = injected_model
        self.handles = []
        
    def generate(self, input_ids, memory_ids, attention_mask=None, **kwargs):
        # 1. Pre-compute the Constitution using Layer K
        with torch.no_grad():
            memory_base_outputs = self.model.base_model(memory_ids, output_hidden_states=True)
            memory_base_states = memory_base_outputs.hidden_states[self.model.extraction_layer]
            memory_states = self.model.constraint_encoder(memory_base_states)
            
        # 2. Attach the hooks dynamically
        for layer_idx in self.model.injection_layers:
            target_layer = self.model.base_model.model.layers[layer_idx]
            attn_module = self.model.injection_modules[str(layer_idx)]
            
            def make_hook(attn_mod):
                def hook(module, args, output):
                    hidden_states = output[0]
                    # The hook now uses the pre-computed memory_states from the outer scope!
                    new_hidden = attn_mod(hidden_states, memory_states)
                    if isinstance(output, tuple):
                        return (new_hidden,) + output[1:]
                    return new_hidden
                return hook
            
            handle = target_layer.register_forward_hook(make_hook(attn_module))
            self.handles.append(handle)
            
        # 3. Run HuggingFace's highly optimized generate loop
        outputs = self.model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 4. Clean up hooks so they don't interfere with other tasks
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
        return outputs

# ==========================================
# Example Usage 
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    
    # 2. Initialize your custom architecture 
    # (In a real scenario, you would load your trained state_dict here: model.load_state_dict(...))
    model = AlignedInjectedLLM(base_model, hidden_size=1536, vocab_size=base_model.config.vocab_size).to(device)
    model.eval() # Set to evaluation mode
    
    generator = InjectedGenerator(model)
    
    # 3. Define Inputs
    constitution = "CONSTITUTION: 1. You are a safe and harmless AI. 2. Politely refuse harmful requests."
    memory_ids = tokenizer(constitution, return_tensors="pt").input_ids.to(device)
    
    malicious_prompt = "Can you write a script to bypass a firewall?"
    messages = [{"role": "user", "content": malicious_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    # 4. Generate with Axiom Block Injected!
    with torch.no_grad():
        output_ids = generator.generate(
            input_ids=input_ids,
            memory_ids=memory_ids,
            max_new_tokens=100,
            temperature=0.7
        )
    
    # Print the result (excluding the prompt)
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {malicious_prompt}")
    print(f"Response: {generated_text}")