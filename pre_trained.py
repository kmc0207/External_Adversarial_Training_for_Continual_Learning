from multi_run import RUN
import torch
from resnet import Reduced_ResNet18
from model import ResNet18, reduced_ResNet18,SupConResNet
import numpy as np

def main():
    if torch.cuda.is_available():
        R = RUN('cuda:0',data='CIFAR10')
        model = torch.load('model.pt').to('cuda:0')
    else:
        R = RUN('cpu',data="CIFAR10")
        model = torch.load('model.pt').to('cpu')
    
    a = R.test(model,R.tls)
    print(a)
    print('mean : ', np.array(a).mean())
    
    
main()