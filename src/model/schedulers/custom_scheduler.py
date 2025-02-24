import torch
from diffusers import DDIMScheduler

class CustomScheduler(DDIMScheduler):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end)