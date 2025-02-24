import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, rank=8):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(768, rank, bias=False)
        self.lora_B = nn.Linear(rank, 768, bias=False)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))