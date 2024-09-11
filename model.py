import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchsummary import summary

class SE(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(SE, self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel*2, in_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self,x):
        ax = self.avepool(x).view(x.size(0), -1)
        mx = self.maxpool(x).view(x.size(0), -1)
        se = torch.concat([ax, mx], dim=1)
        out=self.fc(se)
        out=out.view(out.size(0),out.size(1),1,1)
        return out*x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_simple=1):
        super(BasicBlock, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=down_simple, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.resize = nn.Sequential()
        if down_simple>1 or in_channels!=out_channels:
            self.resize = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=down_simple, stride=down_simple),
                nn.BatchNorm2d(num_features=out_channels),
            )
    def forward(self, x):
        f = self.feature(x)
        x = self.resize(x)
        y = F.leaky_relu(x+f)
        return y

class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_simple=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=down_simple, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.se = SE(in_channel=out_channels, reduction=reduction)
        self.resize = nn.Sequential()
        if down_simple>1 or in_channels!=out_channels:
            self.resize = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=down_simple, stride=down_simple),
                nn.BatchNorm2d(num_features=out_channels),
            )
       
    def forward(self, x):
        f = self.feature(x)
        se = self.se(f)
        x = self.resize(x)
        y = F.leaky_relu(x+se)
        return y
    
class CLPRNet(nn.Module):
    def __init__(self):
        super(CLPRNet, self).__init__()

        self.feature = nn.Sequential(
            BasicBlock(in_channels=3, out_channels=4),
            BasicBlock(in_channels=4, out_channels=16, down_simple=2),
            BasicBlock(in_channels=16, out_channels=16),
            BasicBlock(in_channels=16, out_channels=16),
            BasicBlock(in_channels=16, out_channels=64, down_simple=2),
            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=64),
        )

        self.feature_128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(), 
        )

        self.feature_64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(), 
        )

        self.feature_32 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),   
        )

        self.feature_up_64 = nn.Sequential(
            nn.Conv2d(in_channels=(128+128), out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
        )

        self.feature_up_128 = nn.Sequential(
            nn.Conv2d(in_channels=(64+64), out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
        )

        self.feature_up_256 = nn.Sequential(
            nn.Conv2d(in_channels=(64+64), out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
        )

        self.at_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=9, kernel_size=1),
            nn.Sigmoid(),
        )

        self.detection = nn.Sequential(  
            SEBasicBlock(in_channels=64, out_channels=64, down_simple=2, reduction=2), 
            SEBasicBlock(in_channels=64, out_channels=64, reduction=2),
            SEBasicBlock(in_channels=64, out_channels=128, down_simple=2, reduction=4),
            SEBasicBlock(in_channels=128, out_channels=128, reduction=2),
            SEBasicBlock(in_channels=128, out_channels=128, down_simple=1, reduction=4), 
            SEBasicBlock(in_channels=128, out_channels=128, reduction=2),
        )

        self.detection_head = nn.Sequential(  
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1),
            nn.BatchNorm2d(num_features=5),
            nn.Sigmoid(),
        )

        self.recognition = nn.Sequential(
            SEBasicBlock(in_channels=64, out_channels=64, down_simple=2, reduction=2), 
            SEBasicBlock(in_channels=64, out_channels=128, down_simple=2, reduction=2), 
            SEBasicBlock(in_channels=128, out_channels=256, down_simple=2, reduction=4),
            SEBasicBlock(in_channels=256, out_channels=256, down_simple=2, reduction=4),
        )

        self.recognition_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=73, kernel_size=1),
        )

    
    def forward(self, x):
        x_256 = self.feature(x)
        x_128 = self.feature_128(x_256)   
        x_64 = self.feature_64(x_128) 
        x_32 = self.feature_32(x_64)
        x_up_64 = self.feature_up_64(torch.concat([x_64, F.interpolate(x_32, size=x_64.shape[2:], mode='nearest')], dim=1))
        x_up_128 = self.feature_up_128(torch.concat([x_128, F.interpolate(x_up_64, size=x_128.shape[2:], mode='nearest')], dim=1))
        x_up_256 = self.feature_up_256(torch.concat([x_256, F.interpolate(x_up_128, size=x_256.shape[2:], mode='nearest')], dim=1)) 

        at = self.at_head(x_up_256)
        at_lp = torch.narrow(at, dim=1, start=0, length=1)
        at_ch = torch.narrow(at, dim=1, start=1, length=8)

        y_detection = self.detection(x_256)
        y_detection = self.detection_head(y_detection)

        base = at_lp*0.1
        x_recognition_256_list = []
        for i in range(8):
            x_recognition_256_list.append(x_256*(torch.narrow(at_ch, dim=1, start=i, length=1)+base).expand_as(x_256))
        x_recognition_256 = torch.concat(x_recognition_256_list,dim=0)

        y_recognition = self.recognition(x_recognition_256)
        y_recognition = self.recognition_head(y_recognition)

        y_recognition_list = []
        for i in range(8):
            y_recognition_list.append(torch.narrow(y_recognition, dim=0, start=i*x.size(0), length=x.size(0)))
        y_recognition = torch.concat(y_recognition_list,dim=1)
   
        y_detection = y_detection.transpose(1, 3).transpose(1, 2)
        y_recognition = y_recognition.transpose(1, 3).transpose(1, 2)

        return y_detection, y_recognition, at_lp, at_ch
    

if __name__ == '__main__':

    model = CLPRNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    y_detection, y_recognition, at_lp, at_ch = model(torch.rand((2, 3, 1024, 1024)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    print(y_detection.shape)
    print(y_recognition.shape)
    print(at_lp.shape)
    print(at_ch.shape)

    # summary(model, (3, 1024, 1024))