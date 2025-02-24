import torch
from src.model.custom_attention.flash_attention import FlashAttention

def test_flash_attention():
    attn = FlashAttention()
    hidden_states = torch.randn(1, 10, 768)
    output = attn(None, hidden_states)
    assert output.shape == hidden_states.shape