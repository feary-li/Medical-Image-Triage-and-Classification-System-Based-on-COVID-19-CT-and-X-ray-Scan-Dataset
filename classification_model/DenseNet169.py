import os
import torch
import torchvision
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import myTransforms
from PIL import Image

# 中值滤波函数
# def MedianFilter(img,k=3,padding=None):
#     imarray=np.asarray(img)
#     height = imarray.shape[0]
#     width = imarray.shape[1]
#     if not padding:
#         edge = int((k - 1) / 2)
#         if height - 1 - edge <= edge or width - 1 - edge <= edge:
#             print("The parameter k is to large.")
#             return None
#         new_arr = np.zeros((height, width), dtype="uint8")
#         for i in range(edge,height-edge):
#             for j in range(edge,width-edge):
#                 new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])# 调用np.median求取中值
#     return new_arr

IMG_SIZE = 224
batch_size = 10

# def label_img(img):
#     word_label = img.split('-')[0]
#     if word_label == 'covid': return 0
#     elif word_label == 'noncovid': return 1

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

data_dir = 'content/covid19_CT'

# 监督学习
train_ss = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor(),
    normalize])

test_ss = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor(),
    normalize])

train_ss = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_jh'), transform=train_ss),
        batch_size=batch_size, shuffle=True)

valid_ss = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'validation_jh'), transform=test_ss),
        batch_size=batch_size)

# 实际分类
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    # torchvision.transforms.RandomAffine(degrees=0, shear=(0, 45)),
    # myTransforms.HEDJitter(theta=0.05),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor(),
    normalize])

train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)

valid_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=test_augs),
        batch_size=batch_size)

test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)

## 查看图片
# to_pil_image = torchvision.transforms.ToPILImage()
#
# for image, label in train_iter:
#     img = to_pil_image(image[0])
#     MedianFilter(img)
#     img.show()

# 画图模块
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义准确率计算函数
def accuracy(y_hat, y):
    pred = y_hat.argmax(dim=1)
    res = pred == y
    return res.sum()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), len(y))
    return metric[0] / metric[1]

# 定义预测函数
def predict(net, data_iter, device=None):
    net.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = list(y.numpy())
            y_hat = net(X)
            pred = list(y_hat.argmax(dim=1).cpu().numpy())
            preds.extend(pred)
            targets.extend(y)
    return preds, targets

# 使用GPU
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 定义网络，可选取是否读取预训练模型
finetune_net = torchvision.models.densenet169(pretrained=True) # 读取pytorch中的预训练模型

# 改变最后分类层
finetune_net.classifier = nn.Sequential(
    nn.Linear(finetune_net.classifier.in_features, 512),
    nn.Dropout(p=0.4),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 2)
)
nn.init.xavier_uniform_(finetune_net.classifier[0].weight)
nn.init.xavier_uniform_(finetune_net.classifier[3].weight)
nn.init.xavier_uniform_(finetune_net.classifier[5].weight)

# 定义相关训练函数
def train_batch(net, X, y, loss, trainer, device):
    X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, valid_iter, loss, trainer, scheduler, num_epochs, device=try_gpu()):
    best_acc = 0
    net = net.to(device)
    num_batches = len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'valid acc'])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))

        valid_acc = evaluate_accuracy_gpu(net, valid_iter, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), 'FTdensennet169.params')
        animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc 'f'{metric[1] / metric[3]:.3f}, valid acc {valid_acc:.3f}')


def train_finetuning(net, learning_rate, lr_period, lr_decay, num_epochs=5):
    device = try_gpu(2)  # 这里使用两个 GPU
    loss = nn.CrossEntropyLoss(reduction="none")
    # 这里训练后面的分类层
    trainer = torch.optim.SGD(
        [{'params': net.classifier.parameters()}],
        lr=learning_rate, weight_decay=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    train(net, train_iter, valid_iter, loss, trainer, scheduler, num_epochs, device)

# 监督学习训练函数
def train_s(net, train_ss, valid_ss, loss, trainer, scheduler, num_epochs, device=try_gpu()):
    best_acc = 0
    net = net.to(device)
    num_batches = len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'valid acc'])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_ss):
            l, acc = train_batch(net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))

        valid_acc = evaluate_accuracy_gpu(net, valid_ss, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), 'FTdensennet169_jiandu.params')
        animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc 'f'{metric[1] / metric[3]:.3f}, valid acc {valid_acc:.3f}')


def train_finetuning_s(net, learning_rate, lr_period, lr_decay, num_epochs=5):
    device = try_gpu(2)  # 这里使用两个 GPU
    loss = nn.CrossEntropyLoss(reduction="none")
    # 这里训练后面的分类层
    trainer = torch.optim.SGD(
        [{'params': net.classifier.parameters()}],
        lr=learning_rate, weight_decay=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    train_s(net, train_ss, valid_ss, loss, trainer, scheduler, num_epochs, device)

lr_period, lr_decay, LR = 20, 0.5, 1e-4
train_finetuning_s(finetune_net, LR, lr_period, lr_decay, num_epochs=100)

# 读取监督学习得到的参数，进行迁移学习
finetune_net.load_state_dict(torch.load('FTdensennet169_jiandu.params'))

# 对于实际的分类目标进行二次训练，对参数进行微调
train_finetuning(finetune_net, LR, lr_period, lr_decay, num_epochs=100)

finetune_net.load_state_dict(torch.load('FTdensennet169.params'))

from sklearn.metrics import accuracy_score, classification_report

train_acc = evaluate_accuracy_gpu(finetune_net, train_iter, device = try_gpu(2))
valid_acc = evaluate_accuracy_gpu(finetune_net, valid_iter, device = try_gpu(2))

# 进行测试集的分类
y_pred, test_y = predict(finetune_net, test_iter, try_gpu(2))
# print(test_y)
# print(y_pred)
test_acc = accuracy_score(test_y, y_pred)

# 输出最终结果
print("Train Accuracy:\t", train_acc)
print("Val Accuracy:\t", valid_acc)
print("Test Accuracy:\t", test_acc)
print(classification_report(test_y, y_pred))  
