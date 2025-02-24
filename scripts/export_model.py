import torch
from src.model.diffusion import OptimizedStableDiffusion

def export_model():
    model = OptimizedStableDiffusion(use_flash_attention=True)
    torch.save(model.state_dict(), "models/checkpoints/final_model.pth")