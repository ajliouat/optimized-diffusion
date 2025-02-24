import torch
from src.model.diffusion import OptimizedStableDiffusion
from src.data_loading.dataset import CustomDataset
from torch.utils.data import DataLoader

def train(config):
    # Load dataset
    dataset = CustomDataset(config["dataset"]["path"], image_size=config["dataset"]["image_size"])
    dataloader = DataLoader(dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)

    # Initialize model
    model = OptimizedStableDiffusion(use_flash_attention=config["training"]["use_flash_attention"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch["prompt"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")