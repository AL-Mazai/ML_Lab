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

# 记录训练损失和准确率
train_losses = []
train_accuracies = []
# 记录测试准确率
test_accuracies = []


# 模型训练
def train(epoch):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

        if batch_idx % 300 == 299:  # 每300次迭代输出一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

    # 计算并记录训练准确率
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    print('Train accuracy for epoch {}: {:.2f} %'.format(epoch + 1, train_accuracy))


# 测试
def test():
    model.eval()  # 设置模型为评估模式
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    print("Test accuracy: {:.2f} %".format(test_accuracy))


if __name__ == '__main__':
    start = time.time()
    epoch_num = 20
    for epoch in range(epoch_num):
        train(epoch)
        test()
    end = time.time()
    print('Training time:', end - start)
    plt.figure(figsize=(10, 5))  # 设置图像大小

    # 绘制训练损失、训练准确率和测试准确率
    # plt.plot(range(1, 11), train_losses, label='Train Loss', color='blue')
    plt.plot(range(epoch_num), train_accuracies, label='Train Accuracy', color='orange')
    plt.plot(range(epoch_num), test_accuracies, label='Test Accuracy', color='green')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Loss, Training Accuracy, and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
