from main import Generator, Discriminator
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
lr=0.01
(b1,b2)=0.9,0.999
batch_size=64
n_epoch=200
img_size=(1,80,60)
latent_dim=100
earlystop_patient=5
save_path='D:\study\sugyeong_github\GAN_fake_Finger_vein\model\\'

# Data path설정
train_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\train\\real\\*.jpg'))
valid_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\val\\real\\*.jpg'))
test_files=glob.glob(os.path.join('D:\prlab\class\\2020-1(machin_learning)\data\\test\\real\\*.jpg'))

train_set = DataLoader(Fingervein_Dataset(train_files),batch_size=batch_size,shuffle=True)
valid_set= DataLoader(Fingervein_Dataset(valid_files),batch_size=batch_size,shuffle=True)
test_set=DataLoader(Fingervein_Dataset(test_files),batch_size=batch_size)

# setup
generator = Generator(img_size)
discriminator = Discriminator(img_size)

adversarial_loss = torch.nn.BCELoss()
G_optimizer = torch.optim.Adam(generator.parameters(),lr=lr,betas=(b1,b2))
D_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(b1,b2))

earlystopping = EarlyStopping(patience=earlystop_patient,verbose=True)
val_loss = []
gen_loss=[]
disc_loss=[]
# Train
for epoch in range(n_epoch):
    for i,(imgs,labels) in enumerate(train_set):
        real_labels=labels.view(labels.size(0),1) # [8,1]을 [8]로 만들어줌
        fake_labels= torch.Tensor(imgs.size(0),1).fill_(0.0)
        real_imgs=imgs

        # G와 D를 번걸아가며 훈련함
        generator.train()

        ## train G (G로 만든 이미지 > D가 판별 > Loss계산 > 갱신)

        G_optimizer.zero_grad()
        noise = torch.Tensor(np.random.normal(0, 1, (imgs.size(0), latent_dim)))  # 왜 2차원?
        gen_imgs= generator(noise) #gen_imgs.size() = (8,640,480)
        print(discriminator(gen_imgs))
        g_loss=adversarial_loss(discriminator(gen_imgs),real_labels)
        #loss, 가중치 갱신
        g_loss.backward()
        G_optimizer.step()

        discriminator.train()
        # train_D (real, fake 구별하기)
        D_optimizer.zero_grad()
        d_real_loss = adversarial_loss(discriminator(real_imgs),real_labels)

        d_fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake_labels)
        d_loss=(d_real_loss+d_fake_loss)/2
        #가중치 갱신
        d_loss.backward()
        D_optimizer.step()

        gen_loss.append(g_loss)
        disc_loss.append(d_loss)
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epoch, i, len(train_set), d_loss, g_loss)
        )
    print(
        "******[Epoch %d/%d] [D loss: %f] [G loss: %f]******"
        % (epoch, n_epoch, np.average(gen_loss), np.average(disc_loss))
    )
    # validation
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        for i,(val_imgs,val_labels) in enumerate(valid_set):

            noise=torch.Tensor(np.random.normal(0, 1, (val_imgs.size(0), latent_dim)))
            real_labels=val_labels.view(val_labels.size(0),1)
            fake_labels= torch.Tensor(val_imgs.size(0),1).fill_(0.0)

            gen_valid_imgs=generator(noise)

            val_gloss=adversarial_loss(discriminator(gen_valid_imgs),real_labels)
            val_dloss=(adversarial_loss(discriminator(val_imgs),real_labels)+adversarial_loss(discriminator(gen_valid_imgs),fake_labels))/2
            print("validation loss D{},G{}".format(val_dloss,val_gloss))

            val_loss.append(val_gloss)

            save_image(gen_valid_imgs,"D:\study\sugyeong_github\GAN_fake_Finger_vein\\val_imgs\\"+str(i)+'.jpg')

    valid_loss=np.average(val_loss)
    print("****[validation g_loss %f]****" % valid_loss)
    earlystopping(valid_loss,generator)
    if earlystopping.early_stop:
        print("Early stopping")
        break

# 훈련마친후 모델저장
torch.save(generator.state_dict(),save_path+"generator.pt")
torch.save(discriminator.state_dict(),save_path+"discriminator.pt")