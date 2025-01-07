from train_and_test import train, test
from logger import Logger
from dataset import MyDataset
from config.get_config import get_merge_config
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import LeNet5
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    # 读取参数
    config = get_merge_config()
    batch_size = config.get("batch_size")
    lr = config.get("lr")
    epochs = config.get("epochs")
    checkpoints_path = config.get("checkpoints_path")
    device = config.get("device")
    log_path = config.get("log_path")
    train_image_path = config.get("train_image_path")
    train_label_path = config.get("train_label_path")
    test_image_path = config.get("test_image_path")
    test_label_path = config.get("test_label_path")

    # 构建数据集
    # 定义适用于Tensor的转换操作
    data_transform = transforms.Compose([
        #下面的处理方法适用于PILImage对象，这里读进来的是
        transforms.ToPILImage(),
        # 将图像resize为32*32大小
        transforms.Resize((32, 32)),
        # 转换为tensor (这会将像素值从[0, 255]缩放到[0.0, 1.0])
        transforms.ToTensor(),
        # 使用均值0.1307和标准差0.3081进行归一化
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = MyDataset(train_image_path, train_label_path, transform=data_transform)
    test_dataset = MyDataset(test_image_path, test_label_path, transform=data_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 实例化模型
    model = LeNet5()
    model.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 构造优化器
    # 这里为了实现方便没有采用原文的优化器，提供了SGD和Adam供比对，可以试下两者的效果有什么不同
    optim = optim.SGD(model.parameters(), lr=lr)
    #optim = optim.Adam(model.parameters())
    # 构造日志
    logger = Logger(log_file=log_path)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optim, criterion, epoch, logger)
        test(model, device, test_loader, criterion, logger)
