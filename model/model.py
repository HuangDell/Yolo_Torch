import torch
import torch.nn as nn

from model.base import BaseModel

# 定义yolo模型的架构
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Yolo(BaseModel):

    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_cls=20):

        super(Yolo, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config
        self.body = self._create_yolo_body(self.architecture)
        self.fcs = nn.Sequential(nn.Flatten(), nn.Linear(1024 * split_size * split_size, 496), nn.LeakyReLU(0.1),
                                 nn.Dropout(0.0), nn.Linear(496, split_size * split_size * (num_cls + num_boxes * 5)))

    def forward(self, x):

        return self.fcs(torch.flatten(self.body(x), start_dim=1))

    def _create_yolo_body(self, arch):
        layers = []
        for i in arch:
            if type(i) == tuple:
                layers += [
                    ConvBlock(in_channels=self.in_channels, out_channels=i[1], kernel_size=i[0], stride=i[2],
                              padding=i[3])
                ]
                self.in_channels = i[1]
            elif type(i) == str:
                layers += [
                    nn.MaxPool2d(stride=2, kernel_size=2)
                ]
                # self.in_channels = self.in_channels//2
            elif type(i) == list:
                for j in range(i[-1]):
                    layers += [
                        ConvBlock(in_channels=self.in_channels, out_channels=i[0][1], kernel_size=i[0][0],
                                  stride=i[0][2], padding=i[0][3])
                    ]
                    self.in_channels = i[0][1]
                    layers += [
                        ConvBlock(in_channels=self.in_channels, out_channels=i[1][1], kernel_size=i[1][0],
                                  stride=i[1][2], padding=i[1][3])
                    ]
                    self.in_channels = i[1][1]
        return nn.Sequential(*layers)
