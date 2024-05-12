import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoderDecoder(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling to reduce dimension

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Decoder
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = self.upconv3(x)

        # Optionally apply a final activation, e.g., sigmoid if your output is expected to be [0, 1]
        x = torch.sigmoid(x)

        return x
