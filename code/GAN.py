import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from PIL import Image

class Generator(nn.Module):
    def __init__(self, img_size,latent_dim=100):
        super(Generator,self).__init__()
        self.img_size = img_size
        self.latent_dim=latent_dim
        self.model=nn.Sequential(*self.block(self.latent_dim,128,normalize=False),
                                 *self.block(128,256),
                                 *self.block(256,512),
                                 *self.block(512,1024),
                                 nn.Linear(1024,int(np.prod(self.img_size))), # imgsize 다 곱한거 만큼 출력 (원소간의곱)
                                 nn.Tanh())
    def block(self,in_dim,out_dim,normalize=True):
        layers=[]
        layers.append(nn.Linear(in_dim,out_dim))
        if normalize :
            layers.append(nn.BatchNorm1d(out_dim,0.8)) # 왜0.8인지는 모름 하이퍼파라미터
        layers.append(nn.LeakyReLU(0.2))
        return layers
    def forward(self,z):
        input = z
        gen_img = self.model(input)
        gen_img = gen_img.view(gen_img.size(0),*self.img_size) # 크기확인해보기.
        return gen_img

class Discriminator(nn.Module):
    def __init__(self, img_size, latent_dim=100):
        super(Discriminator, self).__init__()
        self.img_size=img_size
        self.latent_dim=latent_dim
        self.model = nn.Sequential(nn.Linear(int(np.prod(self.img_size)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),)

    def forward(self,img):
        img_flat = img.view(img.size(0), -1)
        print(img_flat.size())
        print(int(np.prod(self.img_size)))
        #print("dis",img_flat.size(),int(np.prod(self.img_size)))
        classify = self.model(img_flat)
        return classify

