from torch.utils.data import Dataset
from array import array
import struct
import torch


class MyDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

        # 读取数据集
        with open(label_path, 'rb') as lbpath:
            # 读取文件的前8个字节（4个字节用于魔数，4个字节用于样本数量）
            # 然后使用struct.unpack解包成两个无符号整数（>II表示大端序的两个32位整数）。
            # 魔数用来确认文件类型和版本，n是标签的数量。
            magic, n = struct.unpack('>II', lbpath.read(8))
            # 继续读取剩下的所有字节，每个字节代表一个标签（0到9之间的数字），并将这些字节存储在一个array对象中
            # 类型码为"B"，表示无符号字符（即0到255）。
            self.labels = array("B", lbpath.read())

        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            self.images = array("B", imgpath.read())
            self.images = torch.ByteTensor(self.images).view(-1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    image_path = "dataset/train-images.idx3-ubyte"
    label_path = "dataset/train-labels.idx1-ubyte"
    my_dataset = MyDataset(image_path, label_path)
    print(my_dataset)
