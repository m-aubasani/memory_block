import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        # Start exactly at 0 so the pre-trained LLM isn't corrupted on step 1
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, memory_states):
        attn_output, _ = self.cross_attn(
            query=hidden_states, key=memory_states, value=memory_states
        )
        # H_new = H_old + tanh(gate) * Constitution_Info
        return hidden_states + torch.tanh(self.gate) * attn_output

class ConstraintEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers=2, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, memory_hidden_states):
        # The input is now the rich tensor from the base LLM's Layer K
        return self.transformer(memory_hidden_states)


class AlignedInjectedLLM(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size,
        layer_pairs=None,
        extraction_layers=None,
        injection_layers=None,
        num_encoder_layers=2,
        num_heads=8,
    ):
        super().__init__()
        self.base_model = base_model

        if layer_pairs is not None:
            self.layer_pairs = [tuple(p) for p in layer_pairs]
        elif extraction_layers is not None and injection_layers is not None:
            if isinstance(extraction_layers, int) and isinstance(injection_layers, int):
                self.layer_pairs = [(extraction_layers, injection_layers)]
            elif isinstance(extraction_layers, int) and isinstance(injection_layers, list):
                self.layer_pairs = [(extraction_layers, inj) for inj in injection_layers]
            elif isinstance(extraction_layers, list) and isinstance(injection_layers, list):
                assert len(extraction_layers) == len(injection_layers), (
                    f"extraction_layers (len {len(extraction_layers)}) and injection_layers "
                    f"(len {len(injection_layers)}) must have the same length"
                )
                self.layer_pairs = list(zip(extraction_layers, injection_layers))
            else:
                raise ValueError("Invalid format for extraction_layers and injection_layers.")
        else:
            raise ValueError("Must provide either layer_pairs or extraction_layers and injection_layers.")
        
        # Dedicated ConstraintEncoder and GatedCrossAttention per layer pair / injection layer
        self.constraint_encoders = nn.ModuleDict({
            str(inj): ConstraintEncoder(
                hidden_size=hidden_size,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
            )
            for ext, inj in self.layer_pairs
        })
        
        self.injection_modules = nn.ModuleDict({
            str(inj): GatedCrossAttention(hidden_size, num_heads=num_heads)
            for ext, inj in self.layer_pairs
        })

    def forward(self, input_ids, attention_mask, memory_ids, labels=None):
        # 1. Pass Constitution through the base LLM to extract hidden states across layers
        with torch.no_grad(): # We don't need gradients for the base model's memory processing
            memory_base_outputs = self.base_model(memory_ids, output_hidden_states=True)

        # 2. Encode memory states separately for each layer pair
        memory_states_by_layer = {}
        for ext_layer, inj_layer in self.layer_pairs:
            raw_mem_states = memory_base_outputs.hidden_states[ext_layer]
            memory_states_by_layer[inj_layer] = self.constraint_encoders[str(inj_layer)](raw_mem_states)
        
        hook_handles = []
        
        # 3. Intercept the hidden states of the base LLM and inject corresponding memory representation
        for ext_layer, inj_layer in self.layer_pairs:
            target_layer = self.base_model.model.layers[inj_layer]
            attn_module = self.injection_modules[str(inj_layer)]
            mem_states = memory_states_by_layer[inj_layer]
            
            def make_hook(attn_mod, mem_st):
                def hook(module, args, output):
                    # Safely handle both tuples and bare tensors
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                        
                    new_hidden_states = attn_mod(hidden_states, mem_st)
                    
                    # Repackage tuple if needed
                    if isinstance(output, tuple):
                        return (new_hidden_states,) + output[1:]
                    return new_hidden_states
                return hook
            
            handle = target_layer.register_forward_hook(make_hook(attn_module, mem_states))
            hook_handles.append(handle)
        
        # 4. Standard Forward Pass
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        
        # 5. Remove hooks to prevent infinite graph memory leaks
        for handle in hook_handles:
            handle.remove()
            
        return outputs