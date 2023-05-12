import torch.cuda
from tqdm import tqdm

from data import load
from model.model import *
from model.yolo_loss import YoloLoss
from util.output import *


def _train(model, train_data, valid_data):
    # torch.cuda.empty_cache()
    optimizer = config.optimizer(model.parameters(), config.lr, weight_decay=config.weight_decay)  # 读取config中设置的优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2,
                                                                     eta_min=5e-5)  # 使用Cos变化的学习率

    loss_list, acc_list = [], []
    criterion = YoloLoss()
    bar = tqdm(range(config.epochs))
    for _ in bar:
        model.train()
        losses = 0.0
        for batch in train_data:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)  # 放在GPU上加速训练
            out = model(inputs)  # Generate predictions
            loss = criterion(out, labels)  # Calculate loss
            loss.backward()
            losses += loss.item()

            nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)  # 截断

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        losses = losses / len(train_data)
        result = f'loss {losses}'

        loss_list.append(losses)
        acc_list.append(0)
        bar.set_description(result)
        printf.info(result)
    return loss_list, acc_list


def test():
    model = Yolo().to(config.device)
    model.load()
    model.eval()
    model.evaluate(te)


def train():
    model = Yolo().to(config.device)
    loss, acc = _train(model, valid_data=load.test_data, train_data=load.train_data)
    save_model(model)
    draw_result(config.epochs, loss, acc)
