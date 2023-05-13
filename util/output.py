import os
import numpy as np
import torch

import config
import matplotlib.pyplot as plt
import shutil
from util.logger import printf
from util.bbox import *

class_dict = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car'
    , 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
              15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}


def create_model_dir():
    os.mkdir(config.path_name)  # 为每个模型创建一个单独的项目保存目录
    os.chdir(config.path_name)  # 将工作目录调整为模型对于的目录


# 输出预测结果
def show_output(label, img, pred):
    img = img[0].permute(1, 2, 0)
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    boxes = cellboxes_to_boxes(pred)[0]
    boxes = nMS(boxes)

    for box in boxes:
        x1, y1, x2, y2 = box_to_corners(box)
        class_name = class_dict[int(box[0]) + 1]
        draw.rectangle([(x1, y1), (x2, y2)], outline="red")
        draw.text((x1, y1 - 30), class_name, fill=(0, 0, 255, 0))

    plt.imshow(img)
    plt.savefig(config.path_name + "images/" + f"{label}.png")


def save_model(model):
    torch.save(model.state_dict(), config.save_name)
    shutil.copy("config.py", config.path_name + "/model.config")  # 将该模型对应的配置信息保存
    printf.info(f"{config.project_name + config.version} save successfully.")


def draw_result(epoch, loss, acc):
    plt.title(config.project_name + config.version)
    plt.plot(range(epoch), loss, label='Train loss')
    plt.plot(range(epoch), acc, label='Test acc')
    plt.legend()
    plt.savefig(config.path_name + config.project_name + ".png")
