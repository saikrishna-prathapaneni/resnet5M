import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from ResNet import ResNetF

# functions to show an image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

def calculate_accuracy(y_pred, y): # calcualting the accuracy of the model
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc



def train(model, iterator, optimizer, criterion, device): # traing the model on with the images in iterator
    
    epoch_loss = 0
    epoch_acc = 0
   
    model.train()
   
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        optimizer.step()
       
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    lr_scheduler.step()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred= model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

if __name__=="__main__":
    transform = transforms.Compose(
    [
     transforms.RandomCrop(size=32, padding=4),
      transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(20),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
     ])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True,transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    model =ResNetF()

    optimizer = optim.SGD(model.parameters(), lr = 1e-1)

    criterion = nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, 200)
    model = model.to(device)
    criterion = criterion.to(device)
  

    EPOCHS = 25
    loss =0
    for i in range(200):
    # using train iterator
        train_loss , epoch_acc = train(model,trainloader, optimizer, criterion, device=device)
        print("train loss per epoch => ", i, train_loss, "train acc per epoch=> " , epoch_acc)
        
        #using validation iterator
        epoch_loss , epoch_valid_acc = evaluate(model, testloader, criterion, device)
        print("val loss per epoch => ", epoch_loss , "val acc per epoch =>",epoch_valid_acc,"\n")
        
        # saving the model whenever there is decrease in loss
        # tracking the model
        if epoch_loss<loss:
            torch.save(model,"./model.pt")
            loss= epoch_loss

