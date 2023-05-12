import torch.nn as nn
import torch
import config
from util.output import *


# 基本模型，包含了训练和测试方法
class BaseModel(nn.Module):
    def __init__(self, is_loaded=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        self.load_state_dict(torch.load(config.save_name, map_location=config.device))

    def train_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(config.device), labels.to(config.device)  # 放在GPU上加速训练
        out = self(inputs)  # Generate predictions
        loss = config.criterion(out, labels)  # Calculate loss
        return loss

    # 测试模型正确率
    def evaluate(self, test_data, test_nums=10):
        with torch.no_grad:
            for i, batch in enumerate(test_data):
                if i == test_nums:
                    return
                inputs, _ = batch
                inputs = inputs.to(config.device)  # 放在GPU上加速训练
                out = self(inputs)
                show_output(out)

