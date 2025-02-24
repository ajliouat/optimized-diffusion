import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from .custom_attention.flash_attention import FlashAttention

class OptimizedStableDiffusion(nn.Module):
    def __init__(self, use_flash_attention=True):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        if use_flash_attention:
            self.pipeline.unet.set_attn_processor(FlashAttention())

    def forward(self, prompt, num_images=1, image_size=512):
        return self.pipeline(prompt, num_images_per_prompt=num_images, height=image_size, width=image_size).images