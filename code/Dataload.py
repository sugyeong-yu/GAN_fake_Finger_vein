import torch
import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class Fingervein_Dataset(Dataset):
    def __init__(self,img_path):
        self.imgs=img_path
        self.labels=torch.Tensor(np.ones(len(img_path)))
    def __len__(self):
        print(len(self.imgs),len(self.labels))
        return len(self.imgs)
    def __getitem__(self, index):
        img= Image.open(self.imgs[index])
        return {'image':img,'label':self.labels[index]}




