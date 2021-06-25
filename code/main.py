import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from PIL import Image

images=glob.glob("D:\prlab\대학원수업\\2020-1(기계학습)\data\\train\\real\\*.jpg")
img=Image.open(images[0])
print(np.array(img).shape)

class GAN(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(GAN, self).__init__()
        self.img_size=img_size
        self.latent_dim=latent_dim

        self.generator=self.build_generator()



    def random_noise(self,mu,sigma):
        noise=torch.Tensor(np.random.normal(0, 1, (self.img_size[0], self.latent_dim))) # 왜 2차원?
        return noise

    def block(self,in_dim,out_dim,normalize=True):
        layers=[]
        layers.append(nn.linear(in_dim,out_dim))
        if normalize :
            layers.append(nn.BatchNorm1d(out_dim,0.8)) # 왜0.8인지는 모름 하이퍼파라미터
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        return layers


    def build_generator(self):
        input = self.random_noise()

        generator = nn.Sequential(*self.block(self.latent_dim,128,normalize=False),
                                 *self.block(128,256),
                                 *self.block(256,512),
                                 *self.block(512,1024),
                                 nn.Linear(1024,int(np.prod(self.img_size))),
                                  nn.Tanh())
        fake_img = generator(input)
        fake_img = fake_img.view(self.img_size(0),self.img_size)
        return fake_img

    def build_discriminator(self,img):







