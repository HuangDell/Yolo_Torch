import shutil
import torch.nn as nn
import torch.optim as optim
import torch
import os

# log information
project_name = "YOLO"
version = "v1.0"
path_name = "out/" + project_name + version + "/"
image_path_name = path_name+"images/"


# model saving
save_name = path_name + project_name + ".pt"

# model parameters
criterion = nn.CrossEntropyLoss()  # 选择交叉熵作为损失函数
optimizer = optim.Adam

# parameters
epochs = 50
lr = 0.001
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data
DATA_PATH = './data/VOC_YOLO'
BATCH_SIZE = 32
CLASS_NUM = 9
WIDTH = 320
HEIGHT = 240
IDX = 1
TEST_NUMS = 30

if not os.path.exists(path_name):
    os.makedirs(path_name)
    os.makedirs(image_path_name)
    shutil.copy("config.py", path_name + "model.config")  # 将该模型对应的配置信息保存
