from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
import os


class RSNADataSet(Dataset):
    '''
    1. opencv对阳性进行图片增强复制*3
    2. 阴性阳性合并
    3. 读取文件构建数据集
    4. 分层划分数据集
    5. 测试集和验证集不进行数据增强, 只转Tesnor
    6. 仅对阳性进行数据增强, 阴性只转Tesnor
    '''

    def __init__(self, is_train, transforms) -> None:
        super().__init__()
        self.is_train = is_train
        self.transforms = transforms
        self.csv = None
        self.folder_path = ''
        if is_train:
            self.csv = pd.read_csv('cancer_detection/data/train.csv')
            self.folder_path = 'cancer_detection/data/train/'
        else:
            self.csv = pd.read_csv('cancer_detection/data/test.csv')
            self.folder_path = 'cancer_detection/data/test/'

    def __getitem__(self, index):
        age = self.csv.loc[index]['age'] / 100.0
        patient_id, image_id = str(self.csv.loc[index]['patient_id']), str(
            self.csv.loc[index]['image_id'])
        img_name = patient_id + '-' + image_id + '.png'
        img_path = os.path.join(self.folder_path, img_name)
        img = Image.open(img_path)

        if self.is_train:
            label = self.csv.loc[index]['cancer']
            if self.transforms is not None:
                if label == 1:
                    img = self.transforms['positive'](img)
                else:
                    img = self.transforms['negative'](img)
            return img, age, label
        else:
            if self.transforms is not None:
                img = self.transforms['negative'](img)
            return img, age

    def __len__(self):
        return len(self.csv)


if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(contrast=[0.5, 2.5]),
        transforms.RandomCrop(256+128),
        transforms.RandomRotation(45),
        transforms.ColorJitter(contrast=[0.5, 2.5]),
        transforms.Resize(size=(512, 512)),
        transforms.Normalize((0.5,), (0.5)),  # 归一化到(-1, 1), 可加可不加
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5)),
    ])

    train_set = RSNADataSet(is_train=True, transforms=train_transforms)
    valid_set = RSNADataSet(is_train=True, transforms=test_transforms)
    test_set = RSNADataSet(is_train=False, transforms=test_transforms)

    print("Dataset is preparing...")
    labels = [train_set[i][2] for i in range(len(train_set))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    train_set = Subset(train_set, train_indices)
    valid_set = Subset(valid_set, valid_indices)
    print("Dataset has finished")
    print(len(train_set))
    print(len(valid_set))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    for batch_id, data in enumerate(valid_loader):
        img, age, label = data
        print(img.shape)
        print(age)
        print(label)
        break
