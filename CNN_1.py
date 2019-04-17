# 　运用ＣＮＮ分析ＭＮＩＳＴ手写数字分类
import numpy as np
import math
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import torch
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from torch import optim
from torchvision import transforms

# 定义CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3),  # 15 26, 26
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2))  # 15, 26,26     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3),  # 48,8,8
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2))  # 48, 4,4

        self.fc = nn.Sequential(
            nn.Linear(10*5*5, 10),
            nn.ReLU(inplace=True))

    def forward(self, x):
        #print(np.shape(x))
        x = self.layer1(x)
        #print(np.shape(x))
        x = self.layer2(x)
        #print(np.shape(x))
        x = x.view(x.size(0), -1)
        #print(np.shape(x))
        x = self.fc(x)

        return x


# 使用内置函数下载mnist数据集
train_set = mnist.MNIST('./data', train=True)
test_set = mnist.MNIST('./data', train=False)

# 预处理=>将各种预处理组合在一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 卷积神经网络
net = CNN()
#print(net)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化策略和学习率
optimizer = optim.SGD(net.parameters(), 1e-1)
# 使用CPU训练，鉴于速度，只迭代二十次
nums_epoch = 20

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for img, label in train_data:
        # img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)

        # 前向传播
        out = net(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img, label in test_data:
        # img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)

        out = net(img)

        loss = criterion(out, label)

        # 记录误差
        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    print('Epoch {} Train Loss {} Train  Accuracy {} Test Loss {} Test Accuracy {}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)))

train_losses_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
train_losses_y=np.array(losses)
plt.plot(train_losses_x,train_losses_y,'r',label='Train Loss')
train_acces_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
train_acces_y=np.array(acces)
plt.plot(train_acces_x,train_acces_y,'g',label='Train Accuracy')

eval_losses_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
eval_losses_y=np.array(eval_losses)
plt.plot(eval_losses_x,eval_losses_y,'b',label='Test loss')
eval_acces_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
eval_acces_y=np.array(eval_acces)
plt.plot(eval_acces_x,eval_acces_y,'y',label='Test Accuacy')

plt.legend()
plt.show()