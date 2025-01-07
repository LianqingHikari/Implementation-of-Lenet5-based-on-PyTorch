from torch.utils.data import Dataset
from array import array
import struct
import torch
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

        # 读取数据集
        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows, cols)
            #self.images = Image.fromarray(self.images)

        with open(label_path, 'rb') as lblpath:
            magic, num = struct.unpack(">II", lblpath.read(8))
            self.labels = np.frombuffer(lblpath.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label
