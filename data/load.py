import pandas as pd
import os

from PIL import Image, ImageDraw
import torchvision.transforms as transform
import torch

import config


class VocDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, label_path, csv, S=7, B=2, C=20, transformation=None):

        super(VocDataset, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.annotations = pd.read_csv(csv)
        self.transform = transformation
        self.img_path = img_path
        self.label_path = label_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        self.label = os.path.join(self.label_path, self.annotations.iloc[index, 1])
        self.img = os.path.join(self.img_path, self.annotations.iloc[index, 0])
        boxes = []
        with open(self.label, 'r') as f:
            for label in f.readlines():
                cls, x, y, width, height = [

                    float(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([cls, x, y, width, height])

        img = Image.open(self.img)
        boxes = torch.tensor(boxes)

        if self.transform:
            img = self.transform(img)

        label_tens = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:

            cls, x, y, width, height = box.tolist()

            i, j = int(self.S * y), int(self.S * x)

            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = self.S * width, self.S * height

            if label_tens[i, j, 20] == 0:
                label_tens[i, j, 20] = 1

                box_coord = torch.tensor([
                    x_cell, y_cell, width_cell, height_cell
                ])

                label_tens[i, j, 21:25] = box_coord

                label_tens[i, j, int(cls)] = 1

        return img, label_tens


tran = transform.Compose([
    transform.Resize((448, 448)),
    transform.ToTensor(),
])

dataset = VocDataset(config.DATA_PATH + '/images', config.DATA_PATH + '/labels', config.DATA_PATH + "/100examples.csv",
                     transformation=tran)

test_data = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
train_data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
