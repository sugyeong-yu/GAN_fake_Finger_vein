from main import Generator, Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset
from Dataload import Fingervein_Dataset
import glob,os

# 파라미터설정
lr=0.001
(b1,b2)=0.9,0.999
batch_size=8
n_epoch=100
img_size=(640, 480)
latent_dim=100

# Data path설정
train_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\train\\real\\*.jpg'))
valid_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\val\\real\\*.jpg'))
test_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\test\\real\\*.jpg'))

train_set = DataLoader(Fingervein_Dataset(train_files),batch_size=batch_size,shuffle=True)
valid_set= DataLoader(Fingervein_Dataset(valid_files),batch_size=batch_size,shuffle=True)
test_set=DataLoader(Fingervein_Dataset(test_files),batch_size=batch_size,shuffle=True)

# setup
generator = Generator(img_size)
discriminator = Discriminator(img_size)

adversarial_loss = torch.nn.BCELoss
G_optimizer = torch.optim.Adam(generator.parameters(),lr=lr,betas=(b1,b2))
D_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(b1,b2))


# Train
for epoch in range(n_epoch):
    for i,(imgs,labels) in enumerate(train_set):
        real_labels=labels
        fake_labels= torch.Tensor(imgs.size(0),1).fill_(0.0)
        real_imgs=imgs

        # G와 D를 번걸아가며 훈련함
        ## train G (G로 만든 이미지 > D가 판별 > Loss계산 > 갱신)

        G_optimizer.zero_grad()
        noise = torch.Tensor(np.random.normal(0, 1, (imgs.size(0), latent_dim)))  # 왜 2차원?
        gen_imgs= generator(noise) #gen_imgs.size() = (8,640,480)

        g_loss=adversarial_loss(discriminator(gen_imgs),real_labels)
        #loss, 가중치 갱신
        g_loss.backward()
        G_optimizer.step()

        # train_D (real, fake 구별하기)
        if i==0:
            exit()
