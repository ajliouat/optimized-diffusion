from src.data_loading.dataset import CustomDataset

def test_data_loading():
    dataset = CustomDataset(root="data/datasets/custom_dataset", image_size=512)
    assert len(dataset) == 2  # Assuming 2 images in the dataset