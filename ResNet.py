import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, skip=True):
        super().__init__()
        self.skip = skip
        self.block = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=kernel_size, padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
           
        )
        if stride == 2 or inchannels != outchannels:
            self.skip = False
            self.skip_conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride,bias=False)
            self.skip_bn = nn.BatchNorm2d(outchannels)
        

    def forward(self, x):
        out = self.block(x)
        if not self.skip:
            out += self.skip_bn(self.skip_conv(x))
        else:
            out += x
        out = F.relu(out.clone())
        return out


class ResNetF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7,stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock1 = ResNet(32, 32,stride=1)
        self.resblock2 = ResNet(32, 64,stride=1)
        self.resblock3 = ResNet(64, 64,stride=1)
        self.resblock4=ResNet(64,64,stride=1)
        self.resblock5=ResNet(64,64,stride=1)
        self.resblock6=ResNet(64,128,stride=2)
        self.resblock7=ResNet(128,128,stride=1)
        self.resblock8=ResNet(128,128,stride=1)
        self.resblock9=ResNet(128,128,stride=1)
        self.resblock10=ResNet(128,128,stride=1)
        self.resblock11=ResNet(128,256,stride=2)
        self.resblock12=ResNet(256,256,stride=1)
        self.resblock13=ResNet(256,256,stride=1)


        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flat=nn.Flatten()
        self.fc1= nn.Linear(in_features=256, out_features=10, bias=True)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x.clone())
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
      

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc1(x) 
     
        return x


model = ResNetF()
model=model.cuda()
random_matrix = torch.rand(1, 3, 224, 224).cuda()
print(model.forward(random_matrix).shape)
summary(model,(3,224,224))