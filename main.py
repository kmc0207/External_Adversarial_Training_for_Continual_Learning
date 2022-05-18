import argparse
import random
import numpy as np
import torch
from multi_run import RUN

def boolean_string(s):
    if s not in {'False','True'}:
        return ValueError('NOt a valid boolean string')
    return s== 'True'

def main(args):
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    R = RUN(args.device,data = args.data,instant=args.instant)
    if args.epoch_control==False:
        R.replay(std_train=args.std_train,vir_train=args.EAT,std_mem=args.std_mem,epoch=args.per_task_epoch,
                 vir_mem=args.vir_mem,mem_size=args.mem_size,lr=args.learning_rate,batch_size=args.batch_size,
                 mem_batch=args.mem_batch_size,subsample=args.mir_subsample,rv=args.rv,instant=args.instant)
    else:
        epoch_control = [args.epoch_1,args.epoch_2,args.epoch_3,args.epoch_4,args.epoch_5]
        R.replay(std_train=args.std_train,vir_train=args.EAT,std_mem=args.std_mem,
                 vir_mem=args.vir_mem,mem_size=args.mem_size,lr=args.learning_rate,batch_size=args.batch_size,
                 mem_batch=args.mem_batch_size,subsample=args.mir_subsample,rv=args.rv,instant=args.instant,
                 epoch_control = epoch_control)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "External Adversarial Attack")
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--batch_size',default=10,type=int)
    parser.add_argument('--mem_batch_size',default=10,type=int)
    parser.add_argument('--mir',default=False,type=boolean_string)
    parser.add_argument('--mir_subsample',default=50,type=int)
    parser.add_argument('--rv',default=False,type=boolean_string)
    parser.add_argument('--per_task_epoch',default = 1,type=int)
    parser.add_argument('--mem_size',default=5000,type=int)
    parser.add_argument('--mem_iter',default=1,type=int)
    parser.add_argument('--data',default='CIFAR100',choices = ['CIFAR10','CIFAR100','miniimagenet'])
    parser.add_argument('--std_train',default=True,type=boolean_string)
    parser.add_argument('--std_mem',default=True,type=boolean_string)
    parser.add_argument('--EAT',default=False,type=boolean_string)
    parser.add_argument('--vir_mem',default=False,type=boolean_string)
    parser.add_argument('--learning_rate',default=0.1,type=float)
    parser.add_argument('--ncm',default=False,type=boolean_string)
    parser.add_argument('--instant',default=True,type=boolean_string)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epoch_control',default=False,type=boolean_string)
    parser.add_argument('--epoch_1',default=1,type=int)
    parser.add_argument('--epoch_2',default=1,type=int)
    parser.add_argument('--epoch_3',default=1,type=int)
    parser.add_argument('--epoch_4',default=1,type=int)
    parser.add_argument('--epoch_5',default=1,type=int)
    args=parser.parse_args()
    main(args)
