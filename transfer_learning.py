import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from custom_dataset import CatsAndDogsDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 3
num_channels = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x
    
dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv',root_dir='cats_dogs_resized',transform=transforms.ToTensor())
train_set,test_set = torch.utils.data.random_split(dataset,[6,4])
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.ReLU(),
                                 nn.Linear(100,10))
model.to(device)
print(model)