# model.py
# Eğitimdeki ile aynı SmallCNN ve preprocess

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

MEAN, STD = 0.1307, 0.3081

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_tensor(img: Image.Image) -> torch.Tensor:
    # gri ton
    img = img.convert("L")
    # arka plan beyazsa invert et (MNIST/EMNIST = siyah zemin üstüne beyaz yazı gibi normalize edilir)
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    # kırp
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)
    # kareye pad + 28x28
    w, h = img.size
    side = max(w, h)
    canv = Image.new("L", (side, side), 0)
    canv.paste(img, ((side - w)//2, (side - h)//2))
    canv = canv.resize((28,28), Image.BILINEAR)

    arr = np.array(canv, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None, :, :]  # (1,1,28,28)
    t = (t - MEAN) / STD
    return t
