from __future__ import  print_function
import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from  torchvision import datasets,transforms,utils
from torch.autograd import Variable
from torchsummary import summary
from PIL import Image

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings("ignore")


from graphs.model.U_net import Unet
from configs.u_net_config import *
from datasets.data_loader import *





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=Unet()

model = model.to(device)


print(summary(model, input_size=(3,224,224)))


optimizer=optim.Adam(model.parameters(),lr=learning_rate)
criterion=nn.BCEWithLogitsLoss()


def train(model,epoch):
    model.train()
    correct=0
    for batch_idx,data in enumerate(train_dataloader):
        data,target=Variable(data['image']),Variable(data['mask'])
        optimizer.zero_grad()
        #print(data.shape)
        output=model.forward(data.float())

        loss=criterion(output.float(),target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.data))


def test(model,epoch):
    model.eval()
    correct=0
    for batch_idx,data in enumerate(test_dataloader):
        data,target=Variable(data['image']),Variable(data['mask'])
        optimizer.zero_grad()
        #print(data.shape)
        output=model.forward(data.float())

        loss=criterion(output.float(),target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                       100. * batch_idx / len(train_dataloader), loss.data))




save_model_name = "unet.pth"
'''
for epoch in range(1,2):
    train(model,epoch)
    test(model,epoch)
   
    torch.save(model, os.path.join(weight_file_path, save_model_name))
'''
from functools import reduce
model=torch.load(os.path.join(weight_file_path, save_model_name))





for batch_idx, data in enumerate(test_dataloader):
    data, target = Variable(data['image']), Variable(data['mask'])
    optimizer.zero_grad()
    # print(data.shape)
    output = model.forward(data.float())
    prediction = F.sigmoid(output)
    prediction=prediction.data.cpu().numpy()


    print(prediction.shape)

    flatten_list = reduce(lambda x, y: x + y, zip(*prediction))

    first_image=flatten_list[2]
    print(first_image.shape)

    plt.imshow(first_image, cmap="gray")
    plt.show()

    break







