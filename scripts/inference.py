from src.model.diffusion import OptimizedStableDiffusion
import yaml

def inference(config):
    # Load model
    model = OptimizedStableDiffusion(use_flash_attention=config["model"]["use_flash_attention"])

    # Generate images
    images = model(config["inference"]["prompt"], num_images=config["inference"]["num_images"], image_size=config["inference"]["image_size"])
    for i, img in enumerate(images):
        img.save(f"output_{i}.png")