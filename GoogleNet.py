import torch
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self,in_channels,out_channels=1000):
        super(self,GoogleNet).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(64,192,kernel_size=3,stride=1)
        self.maxpool2 = nn.Conv2d(kernel_size=3,stride=2)

        self.inception3a = Inception_block(192,64,96,128,16,32,32)
        self.inception3b = Inception_block(256,128,128,192,32,96,64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.inception4a = Inception_block(480,192,96,208,16,48,64)
        self.inception4b = Inception_block(512,160,112,224,24,64,64)
        self.inception4c = Inception_block(512,128,128,256,24,64,64)
        self.inception4d = Inception_block(512,112,144,288,32,64,64)
        self.inception4e = Inception_block(528,256,160,320,32,128,128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.inception5a = Inception_block(832,256,160,320,32,128,128)
        self.inception5b = Inception_block(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(p=0.4)

        self.linear = nn.Linear(1024,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool
        
        x = x.reshape(x.shape[0],-1)

        x = self.dropout(x)

        x = self.linear(x)

        return x

class Inception_block(nn.Module):
    def __init__(self,in_channels,out_1X1,red_3X3,out_3X3,red_5X5,out_5X5,out_1X1pool):
        super(self,Inception_block).__init__()
        self.branch1 = conv_block(in_channels,out_1X1,kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels,red_3X3,kernel_size=3),
            conv_block(red_3X3,out_3X3,kernel_size=3,stride=1,padding=1)
        ) 

        self.branch3 = nn.Sequential(
            conv_block(in_channels,red_5X5,kernel_size=5),
            conv_block(red_5X5,out_5X5,kernel_size=5,stride=1,padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            conv_block(in_channels,out_1X1pool,kernel_size=1,stride=1,padding=1)
        )

    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],1)


class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(self,conv_block).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels,out_channels,*kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.relu(self.batch_norm(self.conv(x)))
    