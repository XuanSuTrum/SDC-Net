# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:29:09 2024

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import re

# 示例数据
file_path = 'E:/所有模型实验结果/paper1/paper1.txt'  # 替换为你的txt文件路径

# 存储解析的数据
epochs_data = []

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        segments = line.split(',')
        epoch_info = {}
        for segment in segments:
            key_value = segment.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                epoch_info[key] = value
        epochs_data.append(epoch_info)
        odd_index_dicts = [epochs_data[i] for i in range(len(epochs_data)) if i % 2 == 1]
# 准备数据
epochs = []
cls_loss = []
transfer_loss = []
train_loss_cmmd_loss = []
train_loss_cls = []
train_loss_cluster = []
total_Loss = []
accuracy = []

for epoch_data in odd_index_dicts:
    epoch = epoch_data["Epoch"]
    epoch_number = int(re.search(r"\d+", epoch).group())
    epochs.append(epoch_number)
    
    cls_loss.append(epoch_data["cls_loss"])
    transfer_loss.append(epoch_data["transfer_loss"])
    train_loss_cmmd_loss.append(epoch_data["train_loss_cmmd_loss"])
    total_Loss.append(epoch_data["total_Loss"])

# 设置横轴刻度，每五个值一个
max_epoch = max(epochs)
x_ticks = np.arange(1, max_epoch + 1, step=5)  # 生成从 1 到最大 epoch 的数组，步长为 5

# 开始绘图，每个loss单独一个图
plt.figure(figsize=(12, 24))

# CLS Loss
plt.subplot(4, 1, 1)
plt.plot(epochs, cls_loss, label='CLS Loss')
plt.xticks(x_ticks)
plt.xlabel('Epoch')
plt.ylabel('CLS Loss')
plt.title('CLS Loss Curve')
plt.legend()
plt.grid(True)

# Transfer Loss
plt.subplot(4, 1, 2)
plt.plot(epochs, transfer_loss, label='Transfer Loss')
plt.xticks(x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Transfer Loss')
plt.title('Transfer Loss Curve')
plt.legend()
plt.grid(True)

# Train CMMD Loss
plt.subplot(4, 1, 3)
plt.plot(epochs, train_loss_cmmd_loss, label='Train CMMD Loss')
plt.xticks(x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Train CMMD Loss')
plt.title('Train CMMD Loss Curve')
plt.legend()
plt.grid(True)


# Total Loss
plt.subplot(4, 1, 4)
plt.plot(epochs, total_Loss, label='Total Loss')
plt.xticks(x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss Curve')
plt.legend()
plt.grid(True)

# 调整布局并显示图形
plt.tight_layout()
plt.show()
