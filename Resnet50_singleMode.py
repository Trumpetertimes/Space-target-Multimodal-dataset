import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import clip

# 一、建立数据集
# animals-6
#   --train
#       |--dog
#       |--cat
#       ...
#   --valid
#       |--dog
#       |--cat
#       ...
#   --test
#       |--dog
#       |--cat
#       ...


# 二、数据增强
# 建好的数据集在输入网络之前先进行数据增强，包括随机 resize 裁剪到 256 x 256，随机旋转，随机水平翻转，中心裁剪到 224 x 224，转化成 Tensor，正规化等。
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


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
                content = int(content)
                self.sam_labels.append(content)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = (self.sam_labels[idx]-1)
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image, torch.tensor(token, dtype=torch.long)

# 三、加载数据
# torchvision.transforms包DataLoader是 Pytorch 重要的特性，它们使得数据增加和加载数据变得非常简单。
# 使用 DataLoader 加载数据的时候就会将之前定义的数据 transform 就会应用的数据上了。
dataset = 'val20_test20'
# train_directory = os.path.join(dataset, 'train')
# valid_directory = os.path.join(dataset, 'valid')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("RN50", device=device, jit=False)
# your_dataset = YourDataset(img_root='/data/STT_dataset/STL/Resnet50_singleMode_label/labels/train/images',
#                            meta_root='./data/STT_dataset/STL/Resnet50_singleMode_label/labels/train',
#                            is_train=True, preprocess=preprocess)
# test_dataset = YourDataset(img_root='/data/STT_dataset/STL/Resnet50_singleMode_label/labels/val/images',
#                            meta_root='./data/STT_dataset/STL/Resnet50_singleMode_label/labels/val',
#                            is_train=True, preprocess=preprocess)
your_dataset = YourDataset(img_root='./data/SIMD_pre_train/Resnet50_single/trian/images',
                           meta_root='./data/SIMD_pre_train/Resnet50_single/trian',
                           is_train=True, preprocess=preprocess)
test_dataset = YourDataset(img_root='./data/SIMD_pre_train/Resnet50_single/test/images',
                           meta_root='./data/SIMD_pre_train/Resnet50_single/test',
                           is_train=True, preprocess=preprocess)
dataset_size_your = len(your_dataset)
test_dataset_size = len(test_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
valid_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
# batch_size = 16
# num_classes = 17
num_classes = 69
# print(train_directory)
# data = {
#     'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#     'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
# }


# train_data_size = len(data['train'])
# valid_data_size = len(data['valid'])
#
# train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=8)
# valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=8)

# print(train_data_size, valid_data_size)

# 四、迁移学习
# 这里使用ResNet-50的预训练模型。
resnet50 = models.resnet50(pretrained=True)


# 在PyTorch中加载模型时，所有参数的‘requires_grad’字段默认设置为true。这意味着对参数值的每一次更改都将被存储，以便在用于训练的反向传播图中使用。
# 这增加了内存需求。由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。
for param in resnet50.parameters():
    param.requires_grad = True

# 为了适应自己的数据集，将ResNet-50的最后一层替换为，将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，接着再连接ReLU层和Dropout层，然后是256 x 72的线性层，输出为72通道的softmax层。
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)

# 用GPU进行训练。
resnet50 = resnet50.to('cuda:0')

# 定义损失函数和优化器。
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

# 五、训练
def train_and_valid(model, loss_function, optimizer, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for inputs, labels in tqdm(your_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for inputs, labels in tqdm(valid_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/dataset_size_your
        avg_train_acc = train_acc/dataset_size_your

        avg_valid_loss = valid_loss/test_dataset_size
        avg_valid_acc = valid_acc/test_dataset_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        if epoch>=epochs:
            torch.save(model, './Resnet_singleMode_models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history

num_epochs = 100
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, './Resnet_singleMode_models/'+dataset+'_'+str(num_epochs)+'.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()
