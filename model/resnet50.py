from torchvision import models

import torch
import torch.nn as nn


class MyResnet(nn.Module):
    def __init__(self, in_channels, class_num) -> None:
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])  # 删除最后的分类层
        self.resnet50.add_module('9', nn.Flatten())  # 添加展平层

        self.age_feature = nn.Sequential(
            nn.Linear(1, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, img, age):
        x = self.resnet50(img)
        # x2 = self.age_feature(age)
        # x = torch.cat([x1, age], 1)
        y = self.classifier(x)

        return y
