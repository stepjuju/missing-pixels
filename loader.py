import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pixelation import pixelate_image

class GrayScaleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = []
        for folder in sorted(os.listdir(image_dir)):
            folder_path = os.path.join(image_dir, folder)
            if os.path.isdir(folder_path):  # ensure that it is a diretiory
                # add all .png from the folder to the list
                self.image_files.extend(
                    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        # Ensure image is a tensor before pixelation
        pixelated_img, mask = pixelate_image(img, percent_pixels=0.1)
        return pixelated_img.float(), mask

    def __len__(self):
        return len(self.image_files)


# Set up transformations and data loading
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

image_dir = r'C:\Users\plani\MyProjects\missing-pixels\data\training_preprocessed'
dataset = GrayScaleImageDataset(image_dir=image_dir, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
