#from GAN import Generator, Discriminator
from DCGAN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,Dataset
from Dataload import Fingervein_Dataset
import glob,os
from torchvision.utils import save_image

img_size=(1,80,60)
batch_size=64
latent_dim=100
nc = 1

# Size of z latent vector (i.e. size of generator input) 잠재벡터의 길이
nz = 100

# Size of feature maps in generator G를 통해 전달되는 feature map의 깊이
ngf = 64

# Size of feature maps in discriminator D를 통해 전달되는 feature map의 깊이
ndf = 64
ngpu=0
test_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\test\\real\\*.jpg'))
test_set=DataLoader(Fingervein_Dataset(test_files),batch_size=batch_size)

generator = Generator(ngpu,nz,ngf,nc)
discriminator = Discriminator(ngpu,nc,ndf)

model_path="D:\study\sugyeong_github\GAN_fake_Finger_vein\\model\\"
generator.load_state_dict(torch.load(model_path+"60-5.313566676819042,5.758914897274667-Generator.pt"))
discriminator.load_state_dict((torch.load(model_path+'60-5.313566676819042,5.758914897274667-Discriminator.pt')))

with torch.no_grad():
    for i,(imgs,labels) in enumerate(test_set):
        generator.eval()
        discriminator.eval()
        noise = torch.randn(imgs.size(0), nz, 1, 1)  # DCGAN쓸때 노이즈
        real_imgs = imgs
        fake_imgs = generator(noise)
        save_image(fake_imgs, "D:\study\sugyeong_github\GAN_fake_Finger_vein\\test_imgs\\" + str(i) + '.jpg')
        real_cls = discriminator(real_imgs)
        print(real_cls)
        fake_cls =  discriminator(fake_imgs)
        print("[real cls]",real_cls)
        print("[fake cls]",fake_cls)