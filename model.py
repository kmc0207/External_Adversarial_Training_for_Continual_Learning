import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os



class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    def all_forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))
        out += self.shortcut(x)
        out = F.relu(out)
        return out1,out
    
class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out
    def all_forward(self, x):
        out1 = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out1,out    


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Reduced_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout=0.3):
        super(Reduced_ResNet, self).__init__()
        self.in_planes = 20
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.layer1 = self._make_layer(block, 20, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 40, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 80, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 160, num_blocks[3], stride=2)
        self.linear = nn.Linear(160*block.expansion, num_classes,bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def features(self, x):
        '''Features before FC layers'''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    def forward_d(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.d1(out)
        out = self.layer2(out)
        out = self.d1(out)
        out = self.layer3(out)
        out = self.d1(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def semi_forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    def all_forward(self,x):
        outs = []
        out = self.conv1(x)
        outs.append(out)
        out = F.relu(self.bn1(out))
        outs.append(out)
        for layer in self.layer1:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer2:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer3:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer4:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        outs.append(out)
        return outs
    def inject_forward(self,x,device):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        size = out.mean()/10
        noise = torch.rand(out.size(),dtype=torch.float)
        noise = noise.to(device)
        noise = noise/noise.mean()
        noise = noise*size
        out = out+noise
        out = self.linear(out)
        return out
    def second_forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear2(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout=0.3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear2 = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def features(self, x):
        '''Features before FC layers'''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    def forward_d(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.d1(out)
        out = self.layer2(out)
        out = self.d1(out)
        out = self.layer3(out)
        out = self.d1(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def semi_forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    def all_forward(self,x):
        outs = []
        out = self.conv1(x)
        outs.append(out)
        out = F.relu(self.bn1(out))
        outs.append(out)
        for layer in self.layer1:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer2:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer3:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        for layer in self.layer4:
            out1,out = layer.all_forward(out)
            outs.append(out1)
            outs.append(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        outs.append(out)
        return outs
    def inject_forward(self,x,device):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        size = out.mean()/10
        noise = torch.rand(out.size(),dtype=torch.float)
        noise = noise.to(device)
        noise = noise/noise.mean()
        noise = noise*size
        out = out+noise
        out = self.linear(out)
        return out
    def second_forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear2(out)
        return out
    
def reduced_ResNet18(num_classes=10):
    return Reduced_ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes)
    
def PreActResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet18(do=0.3,num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes,dropout=do)

def ResNet18_s():
    return ResNet(BasicBlock_s,[2,2,2,2],dropout=0.3)

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='mlp', feat_dim=128,num_classes=100):
        super(SupConResNet, self).__init__()
        self.encoder = reduced_ResNet18(num_classes)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)