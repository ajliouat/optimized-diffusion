import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class CustomDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.root = root
        self.image_size = image_size
        self.transform = Compose([Resize((image_size, image_size)), ToTensor()])
        self.captions = self._load_captions()

    def _load_captions(self):
        with open(os.path.join(self.root, "captions.txt"), "r") as f:
            return [line.strip().split("|") for line in f.readlines()]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path, caption = self.captions[idx]
        img = Image.open(os.path.join(self.root, "images", img_path)).convert("RGB")
        return self.transform(img), caption