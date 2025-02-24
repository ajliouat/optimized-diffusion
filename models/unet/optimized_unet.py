import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class OptimizedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")

    def forward(self, x, timestep, encoder_hidden_states):
        return self.unet(x, timestep, encoder_hidden_states).sample