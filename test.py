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

device = "cuda" if torch.cuda.is_available() else "cpu"


# model, preprocess = clip.load("ViT-B/32", device=device)
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
        return image, token, self.samples[idx], self.sam_labels[idx]


# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("RN50", device=device, jit=False)

# your_dataset = YourDataset(img_root='/data/STT_dataset/STL/train/images',
#                            meta_root='./data/STT_dataset/STL/train',
#                            is_train=True, preprocess=preprocess)
# your_dataset = YourDataset(img_root='/data/STT_dataset/STL/val/images',
#                            meta_root='./data/STT_dataset/STL/val',
#                            is_train=True, preprocess=preprocess)
# your_dataset = YourDataset(img_root='/data/STT_dataset/STL/all_test/images',
#                            meta_root='./data/STT_dataset/STL/all_test',
#                            is_train=True, preprocess=preprocess)
your_dataset = YourDataset(img_root='/data/SIMD_pre_train/val02_test02_train/test/images',
                           meta_root='./data/SIMD_pre_train/val02_test02_train/test',
                           is_train=True, preprocess=preprocess)
dataset_size_your = len(your_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)

# 加载权重文件
# weights_path = "./your model name_epoch_99.pth"  # 替换为你的权重文件路径
weights_path = "./val02_test02_epoch_99.pth"  # 替换为你的权重文件路径
state_dict = torch.load(weights_path)
# 将权重加载到模型中
net.load_state_dict(state_dict)
# 将模型设置为评估模式（如果是推理阶段）
net.eval()


def acc_judge(matrix, text):
    # 输入矩阵
    # matrix = np.array()

    # 初始化正确计数
    correct_count = 0

    # 遍历每个向量
    for i in range(matrix.shape[0]):
        # 获取第 i 个向量
        vector = matrix[i]
        # 找到向量中最大值的索引
        max_index = np.argmax(vector)
        # 判断最大值是否在第 i 个位置
        text_predicted = text[max_index]
        if float(vector[max_index]) == float(vector[i]):
            correct_count += 1

    # 计算平均正确率
    accuracy = correct_count / matrix.shape[0]
    print(f"平均正确率: {accuracy * 100:.2f}%")
    return accuracy, text_predicted

def acc_judge_all(matrix):
    # 初始化正确计数
    correct_count = 0

    # 遍历每个向量
    for i in range(len(matrix)):
        correct_count += float(matrix[i])

    # 计算平均正确率
    accuracy = correct_count / len(matrix)
    print(f"平均正确率: {accuracy * 100:.2f}%")
    return accuracy


with torch.no_grad():
    # image_features = model.encode_image(image) # 将图片进行编码
    # text_features = model.encode_text(text)    # 将文本进行编码
    count = 0
    acc = []
    acc_list = []
    for image, text, image_name, text_name in your_dataloader:
        # 将数据移动到 GPU
        image = image.to(device)  # 将图像数据移动到 GPU
        text = text.to(device)  # 将文本数据移动到 GPU
        logits_per_image, logits_per_text = net(image, text)
        # similarity = (100.0 * logits_per_image @ logits_per_text.T).softmax(dim=-1)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # probs = similarity.cpu().numpy()
        count += 1
        print("Label probs:",
              probs)  # prints: [[0.9927937  0.00421068 0.00299572]] # 图片"CLIP.png"对应"a diagram"的概率为0.9927937
        acc, text_pre = acc_judge(probs, text_name)

        acc_list.append(acc)
        print(text_pre)

        # print(image_name)
        # print(text_name)
    ave_acc = acc_judge_all(acc_list)

    print("average acc: ",
          ave_acc)  # 总的平均精度
