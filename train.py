from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from numpy import *

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import load_data, accuracy
from models import GCN
import Accu_eval
from sklearn.metrics import cohen_kappa_score, f1_score

from sklearn.metrics import confusion_matrix


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=6e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load AAAdata
adj,adj1, adj2, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item()+1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

y1=[]
y2=[]


def calculate_triangle_threshold(loss):
    """
    根据给定的损失值分布计算三角阈值。

    :param losses: 损失值数组
    :return: 动态阈值
    """
    loss = np.atleast_1d(np.asarray(loss)).flatten()
    sorted_losses = np.sort(loss)

    # min_loss_idx = sorted_losses[0]
    # max_loss_idx = sorted_losses[-1]

    # # 如果最大和最小值相同，则返回该值作为阈值（避免除以0）
    # if max_loss_idx == min_loss_idx:
    #     return sorted_losses[max_loss_idx]

    # 计算三角形的斜率
    slope = (sorted_losses[-1] - sorted_losses[0]) / (len(sorted_losses)-1)

    # 对于每一个点，计算与直线的距离，并找到距离最大的点作为阈值
    distances = []
    for i in range(len(sorted_losses)):

        distance = abs(sorted_losses[i] - (slope * i + sorted_losses[len(sorted_losses)-1]))
        distances.append((distance, sorted_losses[i]))

    # 选择具有最大垂直距离的点作为阈值
    _, threshold = max(distances, key=lambda x: x[0])

    return threshold


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    #前向传递
    output = model(features, adj, adj1, adj2)

    # 计算权重
    # 获取训练集的标签
    train_labels = labels[idx_train]

    num_classes = 12  # 根据实际情况调整

    # 统计训练数据中各标签的出现次数
    unique_labels, counts_train = torch.unique(train_labels, return_counts=True)

    # 初始化全零计数数组（长度为总类别数）
    counts = torch.zeros(num_classes, dtype=torch.long)
    counts[unique_labels] = counts_train  # 填充实际出现的类别计数

    # 计算权重（处理未出现的类别）
    epsilon = 1e-8  # 防止除以0的小量
    counts_float = counts.float().clone()  # 避免修改原始数据
    counts_float[counts_float == 0] = epsilon  # 替换0值为epsilon
    weights = 1.0 / counts_float
    weights[counts == 0] = 0.0  # 将未出现类别的权重设为0

    # 归一化权重（可选）
    weights = weights / weights.sum()

    # 确保权重位于正确设备
    weights = weights.to(output.device)

    # 计算加权损失
    loss = F.nll_loss(output[idx_train], train_labels, weight=weights)

    # 计算损失
    # loss = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
    # # acc_train = accuracy(output[idx_train], labels[idx_train])
    # threshold = calculate_triangle_threshold(loss.detach().cpu().numpy())
    #
    # # 将阈值转换为tensor并筛选大于阈值的损失
    # mask = loss > threshold
    #
    # loss_train = loss[mask].mean()

    loss.backward()

    optimizer.step()

    preds_train = output[idx_train].argmax(dim=1)
    acc_train = (preds_train == labels[idx_train]).float().mean()
    kappa_train = cohen_kappa_score(labels[idx_train].cpu().numpy(), preds_train.cpu().numpy())
    loss_train = loss.item()
    f1_train = f1_score(labels[idx_train].cpu().numpy(), preds_train.cpu().numpy(), average='weighted')

    if not args.fastmode:
        model.eval()
        output = model(features, adj, adj1, adj2)

    model.eval()
    with torch.no_grad():
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        preds_val = output[idx_val].argmax(dim=1)
        acc_val = (preds_val == labels[idx_val]).float().mean()
        kappa_val = cohen_kappa_score(labels[idx_val].cpu().numpy(), preds_val.cpu().numpy())
        loss_val = loss_val.item()
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds_val.cpu().numpy(), average='weighted')
    print(
        f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, Train Kappa: {kappa_train:.4f}, Train F1: {f1_train:.4f}, Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, Val Kappa: {kappa_val:.4f}, Val F1: {f1_val:.4f},time:{time.time() - t}")

    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))

    y1.append(acc_val)
    y2.append(loss_val)

    if epoch+1 == args.epochs:
        # 计算并打印混淆矩阵
        cm = confusion_matrix(labels[idx_val].detach().cpu().numpy(), preds_val.detach().cpu().numpy())
        print("Confusion Matrix:\n", cm)


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    preds = output.max(1)[1].type_as(labels)
    # df_labels = pd.DataFrame({'node_id': idx_test, 'label': preds})
    # 保存为CSV文件
    # df_labels.to_csv(f'output_GCN.csv', index=False)
    Accu_eval.p_r(preds.numpy(), labels[idx_test].numpy())
    # # 将所有预测结果进行输出：
    # data1 = pd.read_csv(
    #     r"D:\研究生\利用保留语义的POI嵌入估计城市功能分布\新数据\POI_A.csv",
    #     encoding='gbk')
    # # a = range(0, len(data1['Level2']), 1)
    # # a = torch.LongTensor(a)
    # preds = output.max(1)[1].type_as(labels)
    # data1["pre"] = preds
    # data1['real']=labels
    # data1.to_csv(r"C:\Users\Lenovo\Desktop\训练图\预测类GCN.csv", encoding='gbk', index=False)


# Train model
t_total = time.time()

# 创建画布和子图
def hua(epoch):
    x = range(epoch)
    fig, ax1 = plt.subplots()

    # 绘制第一条折线图
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(x, y1, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)

    # 对 Loss 数据进行压缩，将前四个数据除以一个较大的数，以缩小刻度
    compressed_y2 = [3 if i < 4 else y for i, y in enumerate(y2)]

    ax2.plot(x, compressed_y2, color=color, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # 添加标题
    ax1.set_title('Accuracy and Loss over Epochs')

    # 显示图形
    plt.show()

for epoch in range(args.epochs):
    train(epoch)
    # if epoch>=1000 and (epoch+1)%100 ==0:
    #     hua(epoch+1)


# hua(args.epochs)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
