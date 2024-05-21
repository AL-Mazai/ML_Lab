import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from lab_7.lab_7_model import Net

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 转变为图像张量，变为多通道  1*28*28
    transforms.Normalize((0.1307,), (0.3081,))])  # 归一化 0-1分布

# 导入MNIST训练数据集
train_dataset = datasets.MNIST(
    root="dataset/mnist",
    train=True,
    download=True,
    transform=transform
)
# 创建一个数据加载器，用于迭代训练数据集
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size
)
# 导入MNIST测试数据集
test_dataset = datasets.MNIST(
    root="dataset/mnist",
    train=False,
    download=True,
    transform=transform
)
# 创建一个数据加载器，用于迭代测试数据集
test_loader = DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=batch_size
)

# 定义并实例化模型
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型加载到GPU
model.to(device)
# 定义损失函数为交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# 定义优化器为随机梯度下降（SGD），学习率为0.01，动量为0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 模型训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零、前馈、反馈、更新
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:  # 300次迭代输出一次
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 测试
accuracy = []


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 无梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # _为每行最大值 predicted为最大值下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set:%s %%" % (100 * correct / total))
    accuracy.append(100 * correct / total)


if __name__ == '__main__':
    start = time.time()
    for epoch in range(10):
        train(epoch)
        test()
    end = time.time()
    print('training time:', end - start)
    print(accuracy)
    plt.plot(range(10), accuracy)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.show()
