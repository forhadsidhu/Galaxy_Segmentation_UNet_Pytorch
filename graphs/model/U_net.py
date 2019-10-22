import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
from torch.autograd import Variable


#Will be changed based on input image requirement
start_fm=64



class double_conv(nn.Module):
    '''
                        each followed by a rectiﬁed linear unit (ReLU) and a 2x2 max pooling
                        operation with stride 2 for downsampling.

    '''

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
         super(double_conv,self).__init__()
         self.conv=nn.Sequential(


                   nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True)

          )

    def forward(self,x):
        x=self.conv(x)
        return  x

class Unet(nn.Module):
    '''
    1. __init__ function()
    it contains initialization..
    '''
    def __init__(self):
        super(Unet,self).__init__()


        #first convolution box
        self.double_conv1=double_conv(3,start_fm,3,1,1)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        #second convolution box
        self.double_conv2=double_conv(start_fm,start_fm*2,3,1,1)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        #third convolution box
        self.double_conv3=double_conv(start_fm*2,start_fm*4,3,1,1)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)

        #Fourth convolution box
        self.double_conv4=double_conv(start_fm*4,start_fm*8,3,1,1)
        self.maxpool4=nn.MaxPool2d(kernel_size=2)

        #Fift convolution box
        self.double_conv5=double_conv(start_fm*8,start_fm*16,3,1,1)

        #Fourth transpose box
        '''
         Firstly, transpose krbo,(ager size a back krbo) upsampling
         then convolution
        
        '''
        self.t_conv4=nn.ConvTranspose2d(start_fm*16,start_fm*8,2,2)
        self.ex_double_conv4=double_conv(start_fm*16,start_fm*8,3,1,1)


        #Third transpose box
        self.t_conv3=nn.ConvTranspose2d(start_fm*8,start_fm*4,2,2)
        self.ex_double_conv3=double_conv(start_fm*8,start_fm*4,3,1,1)


        #second transpose box
        self.t_conv2=nn.ConvTranspose2d(start_fm*4,start_fm*2,2,2)
        self.ex_double_conv2=double_conv(start_fm*4,start_fm*2,3,1,1)

        #First transpose box
        self.t_conv1=nn.ConvTranspose2d(start_fm*2,start_fm,2,2)
        self.ex_double_conv1=double_conv(start_fm*2,start_fm,3,1,1)

        #one by one convolution
        self.one_by_one=nn.Conv2d(start_fm,1,1,1,0)



     #lets define forward function

    def forward(self,inputs):

        #coontracting path
        conv1=self.double_conv1(inputs)
        maxpool1=self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Bottom
        conv5 = self.double_conv5(maxpool4)

        #Expanding path
        t_conv4=self.t_conv4(conv5)
        #copy and crop
        cat4=torch.cat([conv4,t_conv4],1)
        ex_conv4=self.ex_double_conv4(cat4)

        # Expanding path
        t_conv3= self.t_conv3(ex_conv4)
        # copy and crop
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        # Expanding path
        t_conv2 = self.t_conv2(ex_conv3)
        # copy and crop
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        # Expanding path
        t_conv1 = self.t_conv1(ex_conv2)
        # copy and crop
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)

        one_by_one=self.one_by_one(ex_conv1)

        return  one_by_one







