import torch
import xformers.ops as xops

class MemoryEfficientAttention:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(context)
        value = attn.to_v(context)

        # Apply Memory-Efficient Attention
        out = xops.memory_efficient_attention(query, key, value)
        return attn.to_out(out)