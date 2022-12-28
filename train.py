import torch
import torch.nn as nn
import numpy as np
import random
import math
import datetime

from torch.utils.data import DataLoader
from torch.optim import Adam
from model.data_set import RSNADataSet
from model.resnet50 import MyResnet
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class ModelConfig:
    def __init__(self) -> None:
        self.algo_name = 'Resnet50'  # 算法名称
        self.env_name = 'Pytorch With cuda11.6'  # 环境名称
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 1
        self.n_epochs = 300
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.img_size = 512
        self.valid_size = 0.2
        self.path = 'cancer_detection/outputs/' + self.algo_name + '/' + curr_time + '/'


def train(cfg):
    model = MyResnet(1, 2).to(cfg.device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    train_transforms = {
        'negative': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize((0.5,), (0.5)),
        ]),
        'positive': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(int(cfg.img_size * 0.75)),
            transforms.RandomRotation(45),
            # transforms.ColorJitter(contrast=[0.5, 2.5]),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize((0.5,), (0.5)),  # 归一化到(-1, 1), 可加可不加
        ])
    }
    test_transforms = {
        'negative': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize((0.5,), (0.5)),
        ]),
        'positive': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize((0.5,), (0.5)),
        ])
    }

    train_set = RSNADataSet(is_train=True, transforms=train_transforms)
    valid_set = RSNADataSet(is_train=True, transforms=test_transforms)

    print("Dataset is preparing...")
    labels = [train_set[i][2] for i in range(len(train_set))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.valid_size, random_state=cfg.seed)  # 使用工具包分层划分数据
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    train_set = Subset(train_set, train_indices)
    valid_set = Subset(valid_set, valid_indices)
    print("Dataset has finished")
    print("Train Data: {}".format(len(train_set)))
    print("Valid Data: {}".format(len(valid_set)))

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, drop_last=False)

    best_eval_acc = -math.inf
    writer = SummaryWriter(cfg.path + 'train_image')
    print('Start Training!')
    print(f'Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}')
    for epoch in range(cfg.n_epochs):
        print("----------epoch: {}----------".format(epoch))
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        model.train()
        for i, data in enumerate(train_loader):
            imgs, ages, labels = data
            ages = ages.float().unsqueeze(-1)
            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()
            imgs, ages, labels = imgs.to(cfg.device), ages.to(cfg.device), labels.to(cfg.device)

            outputs = model(imgs, ages)
            mean_acc = np.float32((torch.max(outputs, 1)[1] == labels).cpu()).mean()
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(mean_acc)
            if i % 20 == 0:
                print("Train Step: {}\tLoss: {:.4f}\tAccuracy: {:.2f}".format(i, loss.item(), mean_acc))

        model.eval()
        for i, data in enumerate(valid_loader):
            imgs, ages, labels = data
            ages = ages.float().unsqueeze(-1)
            imgs, ages, labels = imgs.to(cfg.device), ages.to(cfg.device), labels.to(cfg.device)

            with torch.no_grad():
                outputs = model(imgs, ages)
                mean_acc = np.float32((torch.max(outputs, 1)[1] == labels).cpu()).mean()
                loss = loss_func(outputs, labels)

            eval_loss.append(loss.item())
            eval_acc.append(mean_acc)
            if i % 20 == 0:
                print("Eval Step: {}\tLoss: {:.4f}\tAccuracy: {:.2f}".format(i, loss.item(), mean_acc))

        epoch_train_loss = sum(train_loss) / len(train_loss)
        epoch_train_acc = sum(train_acc) / len(train_acc)
        epoch_eval_loss = sum(eval_loss) / len(eval_loss)
        epoch_eval_acc = sum(eval_acc) / len(eval_acc)
        if epoch_eval_acc >= best_eval_acc:
            torch.save(model.state_dict(), cfg.path + 'best_model')
            best_eval_acc = epoch_eval_acc

        torch.save(model.state_dict(), cfg.path + 'final_model')
        print("----Epoch: {}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.2f}".format(epoch, epoch_train_loss,
                                                                                 epoch_train_acc))
        print("----Epoch: {}\tEval Loss: {:.4f}\tEval Accuracy: {:.2f}".format(epoch, epoch_eval_loss, epoch_eval_acc))
        writer.add_scalars(main_tag='loss',
                           tag_scalar_dict={
                               'train': epoch_train_loss,
                               'eval': epoch_eval_loss
                           },
                           global_step=epoch)
        writer.add_scalars(main_tag='acc',
                           tag_scalar_dict={
                               'train': epoch_train_acc,
                               'eval': epoch_eval_acc
                           },
                           global_step=epoch)

    writer.close()


def set_seed(seed):
    """
    全局生效
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    cfg = ModelConfig()
    set_seed(cfg.seed)
    Path(cfg.path).mkdir(parents=True, exist_ok=True)
    train(cfg=cfg)
