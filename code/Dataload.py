import torch
import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Fingervein_Dataset(Dataset):
    def __init__(self,img_path):
        self.imgs=img_path
        self.labels=torch.Tensor(np.ones(len(img_path)))
    def __len__(self):
        print(len(self.imgs),len(self.labels))
        return len(self.imgs)
    def __getitem__(self, index):
        img= Image.open(self.imgs[index])
        trans= transforms.Compose([transforms.ToTensor()])
        tensor_img=trans(img)
        print(self.imgs[index])
        return tensor_img,self.labels[index]

# file_path='D:\prlab\class\\2020-1(machin_learning)\data\\train\\real\\*.jpg'
# file_list=glob.glob(os.path.join(file_path))
# Fingervein_Dataset(file_list).__getitem__(0)



