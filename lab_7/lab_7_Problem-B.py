import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# 自定义数据集类
class OlivettiFacesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(57, 47).astype(np.float32)  # 将数据调整为正确的形状
        label = self.labels[idx]  # 获取对应的标签
        if self.transform:
            image = self.transform(image)  # 如果定义了转换操作，应用到图像
        return image, label


# 加载数据
def load_data(dataset_path):
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]  # 获取所有图像文件
    image_files.sort()  # 确保文件顺序一致
    faces_data = np.empty((400, 2679))  # 初始化存储图像数据的数组
    for i, image_file in enumerate(image_files):
        img = Image.open(image_file).convert('L')  # 读取图像并转换为灰度图
        img_ndarray = np.asarray(img, dtype='float32') / 255.0  # 转换为numpy数组并归一化
        faces_data[i] = img_ndarray.flatten()  # 将图像展开并存储到数组中

    labels = np.array([i // 10 for i in range(400)])  # 生成对应的标签
    return faces_data, labels


# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=2)  # 第一层卷积，输入通道1，输出通道20，卷积核大小2
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)  # 第二层卷积，输入通道20，输出通道40，卷积核大小3
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，窗口大小2x2
        self.dropout1 = nn.Dropout(0.25)  # 第一个Dropout层，丢弃率0.25
        self.fc1 = nn.Linear(40 * 13 * 10, 1000)  # 全连接层，输入尺寸40*13*10，输出尺寸1000
        self.dropout2 = nn.Dropout(0.5)  # 第二个Dropout层，丢弃率0.5
        self.fc2 = nn.Linear(1000, 40)  # 最后一层全连接层，输出尺寸40
        self.activation = nn.Tanh()  # Tanh激活函数

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))  # 通过第一层卷积和池化
        x = self.pool(self.activation(self.conv2(x)))  # 通过第二层卷积和池化
        x = self.dropout1(x)  # 应用Dropout
        x = x.view(-1, 40 * 13 * 10)  # 展平特征图
        x = self.activation(self.fc1(x))  # 通过全连接层
        x = self.dropout2(x)  # 应用Dropout
        x = self.fc2(x)  # 最后一层全连接层
        return x


# 训练模型
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=35):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到设备（GPU或CPU）

            optimizer.zero_grad()  # 清除梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累加损失

        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


# 测试模型
def test_model(model, test_loader):
    model.load_state_dict(torch.load('model_weights.pth'))  # 加载模型权重
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    # 数据集路径
    dataset_path = 'dataset/faces'
    faces_data, labels = load_data(dataset_path)  # 加载数据

    # 划分数据集为训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(faces_data, labels, test_size=0.2, stratify=labels,
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    # 定义数据转换操作
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 创建数据集对象
    train_dataset = OlivettiFacesDataset(X_train, y_train, transform=transform)
    val_dataset = OlivettiFacesDataset(X_val, y_val, transform=transform)
    test_dataset = OlivettiFacesDataset(X_test, y_test, transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
    # 创建模型，并将其移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)
    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=35)
    # 测试模型
    test_model(model, test_loader)
