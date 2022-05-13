import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as D
import copy
import pickle
import os
from itertools import combinations
import random

class dataset_transform(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.transform = transforms.Compose([transforms.ToTensor()])  # save the transform

    def __len__(self):
        return len(self.y)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        if self.transform:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x.float(), self.y[idx]


    
def setting_data_transform(shuffle=False,give=False):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32,padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2675))
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2675))
        ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False)
    set_x = [[] for i in range(5)]
    set_y = [[] for i in range(5)]
    set_x_ = [[] for i in range(5)]
    set_y_ = [[] for i in range(5)]
    if shuffle==False:
        for batch_images, batch_labels in train_loader:
          if batch_labels >= 5:
            y = batch_labels-5
          else :
            y = batch_labels
          set_x_[y].append(batch_images)
          set_y_[y].append(batch_labels)
        for i in range(5):
          set_x[i] = torch.stack(set_x_[i])
          set_y[i] = torch.stack(set_y_[i])
        set_x_t = [[] for i in range(5)]
        set_y_t = [[] for i in range(5)]
        set_x_t_ = [[] for i in range(5)]
        set_y_t_ = [[] for i in range(5)]
        for batch_images, batch_labels in test_loader:
          if batch_labels >= 5:
            y = batch_labels-5
          else :
            y = batch_labels
          set_x_t_[y].append(batch_images)
          set_y_t_[y].append(batch_labels)
        for i in range(5):
          set_x_t[i] = torch.stack(set_x_t_[i])
          set_y_t[i] = torch.stack(set_y_t_[i])
        ds = torch.load('cifar10_primitive.pt')
        dl = torch.utils.data.DataLoader(ds,batch_size=1)
        set_x_v_ = [[] for i in range(5)]
        set_y_v_ = [[] for i in range(5)]
        set_x_v = [[] for i in range(5)]
        set_y_v = [[] for i in range(5)]
        for image,label in dl:
            y = int(label)
            #print(y.size())
            if y>=5:
                y = label - 5
            else :
                y = label
            set_x_v_[y].append(image)
            set_y_v_[y].append(label)
        for i in range(5):
            set_x_v[i] = torch.stack(set_x_v_[i])
            set_y_v[i] = torch.stack(set_y_v_[i])
    else :
        if give !=False:
            num = give
        else:
            num = [i for i in range(10)]
            random.shuffle(num)
        dic = {}
        for i in range(5):
            dic[num[i]] = i
            dic[num[i+5]] = i
        print(dic)
        for batch_images, batch_labels in train_loader:
            y = int(batch_labels)
            set_x_[dic[y]].append(batch_images)
            set_y_[dic[y]].append(batch_labels)
        for i in range(5):
            set_x[i] = torch.stack(set_x_[i])
            set_y[i] = torch.stack(set_y_[i])
        set_x_t = [[] for i in range(5)]
        set_y_t = [[] for i in range(5)]
        set_x_t_ = [[] for i in range(5)]
        set_y_t_ = [[] for i in range(5)]
        for batch_images, batch_labels in test_loader:
            y = int(batch_labels)
            set_x_t_[dic[y]].append(batch_images)
            set_y_t_[dic[y]].append(batch_labels)
        for i in range(5):
            set_x_t[i] = torch.stack(set_x_t_[i])
            set_y_t[i] = torch.stack(set_y_t_[i])
        ds = torch.load('cifar10_primitive.pt')
        dl = torch.utils.data.DataLoader(ds,batch_size=1)
        set_x_v_ = [[] for i in range(5)]
        set_y_v_ = [[] for i in range(5)]
        set_x_v = [[] for i in range(5)]
        set_y_v = [[] for i in range(5)]
        for image,label in dl:
            y = int(label)
            #print(y,dic[y])
            set_x_v_[dic[y]].append(image)
            set_y_v_[dic[y]].append(label)
        for i in range(5):
            set_x_v[i] = torch.stack(set_x_v_[i])
            set_y_v[i] = torch.stack(set_y_v_[i])
            
            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return set_x,set_y,set_x_t,set_y_t


def setting_data(shuffle= False,give = False,transform=False):
    if transform==True:
        transform_train = transforms.Compose(
        [transforms.RandomCrop(32,padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2675))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2675))
        ])
    else :
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False)
    set_x = [[] for i in range(5)]
    set_y = [[] for i in range(5)]
    set_x_ = [[] for i in range(5)]
    set_y_ = [[] for i in range(5)]
    if shuffle==False:
        for batch_images, batch_labels in train_loader:
          if batch_labels >= 5:
            y = batch_labels-5
          else :
            y = batch_labels
          set_x_[y].append(batch_images)
          set_y_[y].append(batch_labels)
        for i in range(5):
          set_x[i] = torch.stack(set_x_[i])
          set_y[i] = torch.stack(set_y_[i])
        set_x_t = [[] for i in range(5)]
        set_y_t = [[] for i in range(5)]
        set_x_t_ = [[] for i in range(5)]
        set_y_t_ = [[] for i in range(5)]
        for batch_images, batch_labels in test_loader:
          if batch_labels >= 5:
            y = batch_labels-5
          else :
            y = batch_labels
          set_x_t_[y].append(batch_images)
          set_y_t_[y].append(batch_labels)
        for i in range(5):
          set_x_t[i] = torch.stack(set_x_t_[i])
          set_y_t[i] = torch.stack(set_y_t_[i])
        ds = torch.load('cifar10_primitive.pt')
        dl = torch.utils.data.DataLoader(ds,batch_size=1)
        set_x_v_ = [[] for i in range(5)]
        set_y_v_ = [[] for i in range(5)]
        set_x_v = [[] for i in range(5)]
        set_y_v = [[] for i in range(5)]
        for image,label in dl:
            y = int(label)
            #print(y.size())
            if y>=5:
                y = label - 5
            else :
                y = label
            set_x_v_[y].append(image)
            set_y_v_[y].append(label)
        for i in range(5):
            set_x_v[i] = torch.stack(set_x_v_[i])
            set_y_v[i] = torch.stack(set_y_v_[i])
    else :
        if give !=False:
            num = give
        else:
            num = [i for i in range(10)]
            random.shuffle(num)
        dic = {}
        for i in range(5):
            dic[num[i]] = i
            dic[num[i+5]] = i
        print(dic)
        for batch_images, batch_labels in train_loader:
            y = int(batch_labels)
            set_x_[dic[y]].append(batch_images)
            set_y_[dic[y]].append(batch_labels)
        for i in range(5):
            set_x[i] = torch.stack(set_x_[i])
            set_y[i] = torch.stack(set_y_[i])
        set_x_t = [[] for i in range(5)]
        set_y_t = [[] for i in range(5)]
        set_x_t_ = [[] for i in range(5)]
        set_y_t_ = [[] for i in range(5)]
        for batch_images, batch_labels in test_loader:
            y = int(batch_labels)
            set_x_t_[dic[y]].append(batch_images)
            set_y_t_[dic[y]].append(batch_labels)
        for i in range(5):
            set_x_t[i] = torch.stack(set_x_t_[i])
            set_y_t[i] = torch.stack(set_y_t_[i])
        ds = torch.load('cifar10_primitive.pt')
        dl = torch.utils.data.DataLoader(ds,batch_size=1)
        set_x_v_ = [[] for i in range(5)]
        set_y_v_ = [[] for i in range(5)]
        set_x_v = [[] for i in range(5)]
        set_y_v = [[] for i in range(5)]
        for image,label in dl:
            y = int(label)
            #print(y,dic[y])
            set_x_v_[dic[y]].append(image)
            set_y_v_[dic[y]].append(label)
        for i in range(5):
            set_x_v[i] = torch.stack(set_x_v_[i])
            set_y_v[i] = torch.stack(set_y_v_[i])
            
            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return set_x,set_y,set_x_t,set_y_t

def make_test_loaders(set_x_t,set_y_t):
    test_ds1 = torch.utils.data.TensorDataset(set_x_t[0].view(-1,3,32,32),set_y_t[0].view(-1))
    test_loader1 = torch.utils.data.DataLoader(test_ds1,batch_size=100,shuffle=True)
    test_ds2 = torch.utils.data.TensorDataset(set_x_t[1].view(-1,3,32,32),set_y_t[1].view(-1))
    test_loader2 = torch.utils.data.DataLoader(test_ds2,batch_size=100,shuffle=True)
    test_ds3 = torch.utils.data.TensorDataset(set_x_t[2].view(-1,3,32,32),set_y_t[2].view(-1))
    test_loader3 = torch.utils.data.DataLoader(test_ds3,batch_size=100,shuffle=True)
    test_ds4 = torch.utils.data.TensorDataset(set_x_t[3].view(-1,3,32,32),set_y_t[3].view(-1))
    test_loader4 = torch.utils.data.DataLoader(test_ds4,batch_size=100,shuffle=True)
    test_ds5 = torch.utils.data.TensorDataset(set_x_t[4].view(-1,3,32,32),set_y_t[4].view(-1))
    test_loader5 = torch.utils.data.DataLoader(test_ds5,batch_size=100,shuffle=True)
    test_loaders = [test_loader1,test_loader2,test_loader3,test_loader4,test_loader5]
    return test_loaders
   
def setting_data_2():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=1,
                                             shuffle=False)
    set_x = [[] for i in range(2)]
    set_y = [[] for i in range(2)]
    set_x_ = [[] for i in range(2)]
    set_y_ = [[] for i in range(2)]
    for batch_images, batch_labels in train_loader:
        if batch_labels >= 5:
            y = 1
        else :
            y = 0
        set_x_[y].append(batch_images)
        set_y_[y].append(batch_labels)
    for i in range(2):
        set_x[i] = torch.stack(set_x_[i])
        set_y[i] = torch.stack(set_y_[i])
    set_x_t = [[] for i in range(2)]
    set_y_t = [[] for i in range(2)]
    set_x_t_ = [[] for i in range(2)]
    set_y_t_ = [[] for i in range(2)]
    for batch_images, batch_labels in test_loader:
            if batch_labels >= 5:
                y = 1
            else :
                y = 0
            set_x_t_[y].append(batch_images)
            set_y_t_[y].append(batch_labels)
    for i in range(2):
        set_x_t[i] = torch.stack(set_x_t_[i])
        set_y_t[i] = torch.stack(set_y_t_[i])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return set_x,set_y,set_x_t,set_y_t


def setting_data_100():
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32,padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    set_x = [[] for i in range(10)]
    set_y = [[] for i in range(10)]
    set_x_ = [[] for i in range(10)]
    set_y_ = [[] for i in range(10)]
    for batch_images, batch_labels in train_loader:
        labels = batch_labels%10
        set_x_[labels].append(batch_images)
        set_y_[labels].append(batch_labels)
    for i in range(10):
        set_x[i] = torch.stack(set_x_[i])
        set_y[i] = torch.stack(set_y_[i])
    set_x_t = [[] for i in range(10)]
    set_y_t = [[] for i in range(10)]
    set_x_t_ = [[] for i in range(10)]
    set_y_t_ = [[] for i in range(10)]
    for batch_images, batch_labels in test_loader:
        labels = batch_labels%10
        set_x_t_[labels].append(batch_images)
        set_y_t_[labels].append(batch_labels)
    for i in range(10):
        set_x_t[i] = torch.stack(set_x_t_[i])
        set_y_t[i] = torch.stack(set_y_t_[i])
    tls = []
    for i in range(10):
        ds_t = torch.utils.data.TensorDataset(set_x_t[i].view(-1,3,32,32),set_y_t[i].view(-1))
        tl = torch.utils.data.DataLoader(ds_t,batch_size=100,shuffle=False)
        tls.append(tl)
    return set_x,set_y,tls

def setting_data_100_20():
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32,padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data',train=True,download=True,transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data',train=False,download=True,transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=1,shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
    set_x = [[] for i in range(20)]
    set_y = [[] for i in range(20)]
    set_x_ = [[] for i in range(20)]
    set_y_ = [[] for i in range(20)]
    for batch_images, batch_labels in train_loader:
        labels = batch_labels%20
        set_x_[labels].append(batch_images)
        set_y_[labels].append(batch_labels)
    for i in range(20):
        set_x[i] = torch.stack(set_x_[i])
        set_y[i] = torch.stack(set_y_[i])
    set_x_t = [[] for i in range(20)]
    set_y_t = [[] for i in range(20)]
    set_x_t_ = [[] for i in range(20)]
    set_y_t_ = [[] for i in range(20)]
    for batch_images, batch_labels in test_loader:
        labels = batch_labels%20
        set_x_t_[labels].append(batch_images)
        set_y_t_[labels].append(batch_labels)
    for i in range(20):
        set_x_t[i] = torch.stack(set_x_t_[i])
        set_y_t[i] = torch.stack(set_y_t_[i])
    tls = []
    for i in range(20):
        ds_t = torch.utils.data.TensorDataset(set_x_t[i].view(-1,3,32,32),set_y_t[i].view(-1))
        tl = torch.utils.data.DataLoader(ds_t,batch_size=100,shuffle=False)
        tls.append(tl)
    return set_x,set_y,tls

def create_task_composition(class_nums, num_tasks, fixed_order=False):
    classes_per_task = class_nums // num_tasks
    total_classes = classes_per_task * num_tasks
    label_array = np.arange(0, total_classes)
    if not fixed_order:
        np.random.shuffle(label_array)

    task_labels = []
    for tt in range(num_tasks):
        tt_offset = tt * classes_per_task
        task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
        #print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
    return task_labels


def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y == i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]


def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]

def shuffle_data(x, y):
    perm_inds = np.arange(0, x.shape[0])
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y


def setting_data_image():
    TEST_SPLIT = 1/6
    train_in = open("./mini-imagenet/mini-imagenet-cache-train.pkl", "rb")
    train = pickle.load(train_in)
    train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
    val_in = open("./mini-imagenet/mini-imagenet-cache-val.pkl", "rb")
    val = pickle.load(val_in)
    val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
    test_in = open("./mini-imagenet/mini-imagenet-cache-test.pkl", "rb")
    test = pickle.load(test_in)
    test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
    all_data = np.vstack((train_x, val_x, test_x))
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    print(all_data.shape)
    for i in range(len(all_data)):
        cur_x = all_data[i]
        cur_y = np.ones((600,)) * i
        rdm_x, rdm_y = shuffle_data(cur_x, cur_y)
        x_test = rdm_x[: int(600 * TEST_SPLIT)]
        y_test = rdm_y[: int(600 * TEST_SPLIT)]
        x_train = rdm_x[int(600 * TEST_SPLIT):]
        y_train = rdm_y[int(600 * TEST_SPLIT):]
        train_data.append(x_train)
        train_label.append(y_train)
        test_data.append(x_test)
        test_label.append(y_test)
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)
    task_labels = create_task_composition(class_nums=100, num_tasks=20,
                                           fixed_order=True)
    print(task_labels)
    test_set = []
    for labels in task_labels:
        x_test, y_test = load_task_with_labels(test_data, test_label, labels)
        test_set.append((x_test,y_test))
    x = []
    y = []
    for cur_task in range(20):
        labels = task_labels[cur_task]
        x_train, y_train = load_task_with_labels(train_data, train_label, labels)
        x.append(x_train)
        y.append(y_train)
    tls = []
    for t in test_set:
        test_ds = dataset_transform(t[0],t[1])
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 100, shuffle=True)
        tls.append(test_loader)
            
    return x,y,tls