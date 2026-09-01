import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        # Start exactly at 0 so the pre-trained LLM isn't corrupted on step 1
        self.gate = nn.Parameter(torch.tensor([-10.0]))

    def forward(self, hidden_states, memory_states):
        attn_output, _ = self.cross_attn(
            query=hidden_states, key=memory_states, value=memory_states
        )
        # H_new = H_old + sigmoid(gate) * Constitution_Info
        return hidden_states + torch.sigmoid(self.gate) * attn_output

class ConstraintEncoder(nn.Module):
    # Removed vocab_size, it now directly accepts hidden states
    def __init__(self, hidden_size, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, memory_hidden_states):
        # The input is now the rich tensor from the base LLM's Layer K
        return self.transformer(memory_hidden_states)


class AlignedInjectedLLM(nn.Module):
    # Added extraction_layer (e.g., K=8)
    def __init__(self, base_model, hidden_size, extraction_layer=8, injection_layers=[8, 16]):
        super().__init__()
        self.base_model = base_model
        self.extraction_layer = extraction_layer
        self.injection_layers = injection_layers
        
        # Initialized without vocab_size
        self.constraint_encoder = ConstraintEncoder(hidden_size)
        
        self.injection_modules = nn.ModuleDict({
            str(layer): GatedCrossAttention(hidden_size) for layer in injection_layers
        })

    def forward(self, input_ids, attention_mask, memory_ids, labels=None):
        # 1. Pass Constitution through the first K layers of the frozen LLM
        with torch.no_grad(): # We don't need gradients for the base model's memory processing
            # output_hidden_states=True returns a tuple where index 0 is embeddings, index 1 is layer 1, etc.
            memory_base_outputs = self.base_model(memory_ids, output_hidden_states=True)
            memory_base_states = memory_base_outputs.hidden_states[self.extraction_layer]

        # 2. Pass those rich representations into our small transformer
        memory_states = self.constraint_encoder(memory_base_states)
        
        hook_handles = []
        
        # 2. Intercept the hidden states of the base LLM
        for layer_idx in self.injection_layers:
            target_layer = self.base_model.model.layers[layer_idx]
            attn_module = self.injection_modules[str(layer_idx)]
            
            def make_hook(attn_mod):
                def hook(module, args, output):
                    # Safely handle both tuples and bare tensors
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                        
                    new_hidden_states = attn_mod(hidden_states, memory_states)
                    
                    # Repackage tuple if needed
                    if isinstance(output, tuple):
                        return (new_hidden_states,) + output[1:]
                    return new_hidden_states
                return hook
            
            handle = target_layer.register_forward_hook(make_hook(attn_module))
            hook_handles.append(handle)
        
        # 3. Standard Forward Pass
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        
        # 4. Remove hooks to prevent infinite graph memory leaks
        for handle in hook_handles:
            handle.remove()
            
        return outputs