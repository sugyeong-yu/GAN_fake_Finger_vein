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
(b1,b2)=1,2
batch_size=8
n_epoch=100
img_size=(640, 480)

# Data path설정
file_path='D:\prlab\class\\2020-1(machin_learning)\data\\train\\real\\*.jpg'
file_list=glob.glob(os.path.join(file_path))

dataloader = DataLoader(Fingervein_Dataset(file_list),batch_size=batch_size,shuffle=True)

# setup
generator = Generator(img_size)
discriminator = Discriminator(img_size)

adversarial_loss = torch.nn.BCELoss
G_optimizer = torch.optim.Adam(generator.parameters(),lr=lr,betas=(b1,b2))
D_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(b1,b2))


# Train
for epoch in range(n_epoch):
    for i,(imgs,labels) in enumerate(dataloader):
        fake= torch.Tensor(imgs.size(0),1).fill_(1.0)
        print("f",fake)
        if i==0:
            exit()
