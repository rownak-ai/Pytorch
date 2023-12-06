import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
    
def save_checkpoint(state,filename='mycheckpoint.pth.tar'):
    print('=> Saving_Checkpoint')
    torch.save(state,filename)

def load_checkpoint(checkpoint):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['optimizer'])
    
in_channels = 1
num_classes = 10
learning_rate = 1e-4
batch_size = 1024
num_epochs = 10
load_model = True

train_dataset = datasets.MNIST(root='dataset/', train=True,transform=transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

model = CNN(in_channels=1,num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

for epoch in range(num_epochs):
    losses = []
    if epoch%3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Loss average over epoch {epoch} is {sum(losses)/len(losses):.3f}')

def check_accuracy(loader,model):
    if loader.dataset.train:
        print('Checking training accuracy')
    else:
        print('Checking testing accuracy')
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            score = model(x)
            _,prediction = score.max(1)
            num_correct += (prediction==y).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train() 


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
