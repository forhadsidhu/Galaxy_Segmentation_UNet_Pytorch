
from __future__ import print_function,division
import os
import torch
import cv2 as cv
import  pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from configs.u_net_config import *
from torch.utils import data
from numpy import expand_dims
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class GalaxyDataset(Dataset):

    def __init__(self,dataset_path,mask_image_path,transform=None):
        self.samples=[]


        self.image_path=dataset_path
        self.mask_path=mask_image_path
        self.transform=transform
        for image in os.listdir(self.image_path):
            self.samples.append(image)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        image_name=os.path.join(self.image_path,self.samples[idx])

        image=io.imread(image_name)
        #print(image.shape)

        image = cv.resize(image, (224,224))

        image=image.transpose((-1,0,1))

        mask_image=io.imread(os.path.join(self.mask_path,self.samples[idx]))
        mask_image = cv.resize(mask_image, (224, 224))

        mask_image=expand_dims(mask_image,axis=0)

        #print(mask_image.shape)

        sample={'image':image,'mask':mask_image}

        if self.transform:
            sample=self.transform(sample)
        return  sample





galaxy_dataset=GalaxyDataset(dataset_path=dataset_path,mask_image_path=mask_path,transform=None)
train_dataloader=data.DataLoader(galaxy_dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_worker)

galaxy_test_dataset=GalaxyDataset(dataset_path=test_dataset_path,mask_image_path=test_mask_path,transform=None)
test_dataloader=data.DataLoader(galaxy_test_dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_worker)

