import torch
from src.model.diffusion import OptimizedStableDiffusion

def test_model():
    model = OptimizedStableDiffusion(use_flash_attention=True)
    assert model is not None