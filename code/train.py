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
from pytorchtools import EarlyStopping


# 파라미터설정
lr=0.05
(b1,b2)=0.9,0.999
batch_size=64
n_epoch=150
img_size=(1,64,64)
latent_dim=100
earlystop_patient=1000
save_path='D:\study\sugyeong_github\GAN_fake_Finger_vein\model\\'

## DCGAN 파라미터
workers = 2
# Number of channels in the training images. For color images this is 3
# 입력이미지의 색상채널수
nc = 1

# Size of z latent vector (i.e. size of generator input) 잠재벡터의 길이
nz = 100

# Size of feature maps in generator G를 통해 전달되는 feature map의 깊이
ngf = 64

# Size of feature maps in discriminator D를 통해 전달되는 feature map의 깊이
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0


# Data path설정
train_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\train\\real\\*.jpg'))
valid_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\val\\real\\*.jpg'))


train_set = DataLoader(Fingervein_Dataset(train_files),batch_size=batch_size,shuffle=True)
valid_set= DataLoader(Fingervein_Dataset(valid_files),batch_size=batch_size,shuffle=True)


# setup
# generator = Generator(img_size)
# discriminator = Discriminator(img_size)
generator =  Generator(ngpu,nz,ngf,nc)
discriminator= Discriminator(ngpu,nc,ndf)
generator.apply(weights_init)
discriminator.apply(weights_init)


adversarial_loss = torch.nn.BCELoss()
G_optimizer = torch.optim.Adam(generator.parameters(),lr=lr,betas=(beta1,0.999))
D_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(beta1,0.999))

earlystopping = EarlyStopping(patience=earlystop_patient,verbose=True)
val_gen_loss = []
val_disc_loss = []
gen_loss=[]
disc_loss=[]
# Train
for epoch in range(n_epoch):
    for i,(imgs,labels) in enumerate(train_set):
        real_labels=labels.view(labels.size(0),1) # [8,1]을 [8]로 만들어줌
        fake_labels= torch.Tensor(imgs.size(0),1).fill_(0.0)
        real_imgs=imgs
        # G와 D를 번걸아가며 훈련함
        discriminator.train()
        generator.train()

        #noise = torch.Tensor(np.random.normal(0, 1, (imgs.size(0), latent_dim)))  # 왜 2차원?
        noise = torch.randn(imgs.size(0), nz, 1, 1) # DCGAN쓸때 노이즈
        gen_imgs = generator(noise)  # gen_imgs.size() = (8,640,480)
        ## train G (G로 만든 이미지 > D가 판별 > Loss계산 > 갱신)
        G_optimizer.zero_grad()

        g_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
        # loss, 가중치 갱신
        g_loss.backward()
        G_optimizer.step()

        # train_D (real, fake 구별하기)
        D_optimizer.zero_grad()
        d_real_loss = adversarial_loss(discriminator(real_imgs),real_labels)
        d_fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake_labels)
        d_loss=(d_real_loss+d_fake_loss)/2
        #가중치 갱신
        d_loss.backward()
        D_optimizer.step()

        gen_loss.append(g_loss.item())
        disc_loss.append(d_loss.item())
        print("Classification: [real_cls %f],[fake_cls %f]"%(discriminator(real_imgs).mean(),discriminator(gen_imgs.detach()).mean()))
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: real- %f,fake- %f] [G loss: %f]"
            % (epoch, n_epoch, i, len(train_set), d_real_loss.item(),d_fake_loss.item(), g_loss.item())
        )
    print(
        "******[Epoch %d/%d] [D loss: %f] [G loss: %f]******"
        % (epoch, n_epoch, np.average(disc_loss),np.average(gen_loss))
    )
    # validation
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for i,(val_imgs,val_labels) in enumerate(valid_set):

            #noise=torch.Tensor(np.random.normal(0, 1, (val_imgs.size(0), latent_dim)))
            noise = torch.randn(val_imgs.size(0), nz, 1, 1)  # DCGAN쓸때 노이즈
            real_labels=val_labels.view(val_labels.size(0),1)
            fake_labels= torch.Tensor(val_imgs.size(0),1).fill_(0.0)

            gen_valid_imgs=generator(noise)

            val_gloss=adversarial_loss(discriminator(gen_valid_imgs),real_labels)
            val_dloss=(adversarial_loss(discriminator(val_imgs),real_labels)+adversarial_loss(discriminator(gen_valid_imgs),fake_labels))/2
            print("validation loss D-{},G-{}".format(val_dloss.item(),val_gloss.item()))

            val_gen_loss.append(val_gloss.item())
            val_disc_loss.append(val_dloss.item())
            save_image(gen_valid_imgs,"D:\study\sugyeong_github\GAN_fake_Finger_vein\\val_imgs\\"+str(i)+str(epoch)+'.jpg')

    valid_gen_loss=np.average(val_gen_loss)
    valid_disc_loss=np.average(val_disc_loss)
    print("****[epoch %d validation g_loss %f]****" % (epoch,valid_gen_loss))
    print("****[epoch %d validation d_loss %f]****" % (epoch,valid_disc_loss))

    # loss decrease하면 model저장
    earlystopping(valid_gen_loss,generator,discriminator,path=save_path+str(epoch)+"-"+str(valid_gen_loss)+","+str(valid_disc_loss))
    if earlystopping.early_stop:
        print("Early stopping")
        break
