from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def get_transforms(image_size=512):
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])