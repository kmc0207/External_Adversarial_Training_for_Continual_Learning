from multi_run import RUN
import torch
import numpy as np

def main():
    if torch.cuda.is_available():
        R = RUN('cuda:0',data='CIFAR10')
    else:
        R = RUN('cpu',data="CIFAR10")
    model = torch.load('model.pt')
    a = R.test(model,R.tls)
    print(a)
    print('mean : ', np.array(a).mean())