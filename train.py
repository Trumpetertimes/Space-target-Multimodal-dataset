import os
from turtle import st

from PIL import Image
import numpy as np
import clip
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn


class YourDataset(Dataset):
    def __init__(self, img_root, meta_root, is_train, preprocess):
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        self.meta_root = meta_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        self.train_set_file = os.path.join(meta_root, 'images')
        self.test_set_file = os.path.join(meta_root, 'text')
        self.train_set_text = os.path.join(meta_root, 'text')
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        self.file_name_list = ''
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        # 5.1 训练还是测试数据集
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file
        # 5.2 获得所有的样本(根据自己的情况更改)
        # with open(self.read_file, 'r') as f:
        #     for line in f:
        #         img_path = os.path.join(self.img_root, line.strip() + '.png')
        #         label = line.strip().split('/')[0]
        #         label = label.replace("_", " ")
        #         label = "photo if " + label
        #         self.samples.append(img_path)
        #         self.sam_labels.append(label)
        # 转换为token
        self.file_name_list = os.listdir(self.read_file)
        self.file_name_list_text = os.listdir(self.train_set_text)
        for i in range(len(self.file_name_list)):
            self.samples.append(self.train_set_file + '/' + self.file_name_list[i])
            # 将图片对应的文本文件夹中的内容加载出来
            text_file_name = self.file_name_list[i][:-4] + '.txt'
            with open(meta_root + '/text/' + text_file_name, 'r', encoding='utf-8') as f:
                content = f.readline()
                self.sam_labels.append(content)
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image, token


# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("RN50", device=device, jit=False)

optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1)

# 创建损失函数
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# 加载数据集
your_dataset = YourDataset(img_root='/data/SIMD_pre_train/val02_test02_train/trian/images',
                           meta_root='./data/SIMD_pre_train/val02_test02_train/trian',
                           is_train=True, preprocess=preprocess)
dataset_size_your = len(your_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=40, shuffle=True, num_workers=0, pin_memory=False)
test_dataset = YourDataset(img_root='/data/SIMD_pre_train/val02_test02_train/val/images',
                           meta_root='./data/SIMD_pre_train/val02_test02_train/val',
                           is_train=True, preprocess=preprocess)
dataset_size_test = len(your_dataset)
test_dataloader = DataLoader(your_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

phase = "train"
model_name = "RN50_val20_val20"
ckt_gap = 50
epoches = 100
for epoch in range(epoches):
    scheduler.step()
    total_loss = 0
    batch_num = 0
    # 使用混合精度，占用显存更小
    with torch.cuda.amp.autocast(enabled=True):
        for images, label_tokens in your_dataloader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss += cur_loss
                if phase == "train":
                    cur_loss.backward()
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        clip.model.convert_weights(net)
            if batch_num % 4 == 0:
                logger.info('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
        epoch_loss = total_loss / dataset_size_your
        # torch.save(net.state_dict(), f"{model_name}_epoch_{epoch}.pth")
        # logger.info(f"weights_{epoch} saved")
        if epoch % ckt_gap == 0:
            checkpoint_path = f"{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
        logger.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
        if epoch == (epoches-1):
            torch.save(net.state_dict(), f"{model_name}_epoch_{epoch}.pth")
            logger.info(f"weights_{epoch} saved")
