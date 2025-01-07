import torch

def train(model, device, train_loader, optimizer, critierion, epoch, logger):
    model.train()
    running_loss = 0.0
    for index, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = critierion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if index % 100 == 99:
            logger.info(
                f"Train Epoch: {epoch} [{index * len(img)}/{len(train_loader.dataset)} ({100. * index / len(train_loader):.0f}%)]\tLoss: {running_loss / 100:.6f}")
            running_loss=0.0
def test(model, device, test_loader, criterion, logger):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
