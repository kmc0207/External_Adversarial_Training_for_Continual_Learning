from model import ResNet18, reduced_ResNet18,SupConResNet
from data import setting_data,make_test_loaders,setting_data_100,setting_data_image,dataset_transform,setting_data_100_20,setting_data_transform
from tsne import Analysis
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from Memory import Memory
import torch
import copy
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from training import Trainer,LinfPGDAttack
import numpy as np
from tsne import Analysis
from training import Trainer
import os
#from resnet import Reduced_ResNet18
#from SCL import SupConLoss


class RUN(object):
    def __init__(self,device,a_iter=7,eps=0.0314,alpha=0.00784,data=10,v=1,shuffle=False,give = False,transform=False,instant=True):
        torch.cuda.empty_cache()
        xv=[]
        if data == 'CIFAR10':
            self.data=10
            set_x,set_y,set_x_t,set_y_t= setting_data(shuffle=shuffle,give=give)
            if transform==True:
                set_x_a,set_y_a,set_x_t,set_y_t = setting_data_transform()
                self.set_x_na = set_x
                self.set_y_na = set_y
                set_x = set_x_a
                set_y = set_y_a
                self.transform=True
            test_loaders = make_test_loaders(set_x_t,set_y_t)
            if instant == False:
                xv = torch.load('ciaf10_version_'+str(v)+'.pt')
            self.size=32
        elif data=='CIFAR100':
            self.data=100
            set_x,set_y,test_loaders = setting_data_100_20()
            if instant==False:
                xv = torch.load('cifar100_version_'+str(v))
            self.size=32
        elif data == 'miniimagenet':
            self.data=data
            set_x_np,set_y_np,test_loaders = setting_data_image()
            if instant==False:
                xv = torch.load('image_version_'+str(v))
            set_x = []
            set_y = []
            set_vy = []
            self.size=84
            for i in range(len(set_x_np)):
                ds = dataset_transform(set_x_np[i],set_y_np[i])
                set_vy.append(torch.from_numpy(set_y_np[i]).type(torch.LongTensor))
                dl = D.DataLoader(ds,batch_size=1,shuffle=False)
                x_torch = []
                y_new = []
                for x,y in dl:
                    x_torch.append(x)
                    y_new.append(y)
                #print(y)
                new_x = torch.stack(x_torch)
                new_y = torch.stack(y_new)
                set_y.append(new_y.view(-1))
                set_x.append(new_x)
            self.yv = set_vy
        self.v = v
        self.transform = transform
        self.device = device
        self.xv = xv
        self.set_x = set_x
        self.set_y = set_y
        self.tls = test_loaders
        self.a_iter = a_iter
        self.eps = eps
        if self.data == '100_20':
            self.data = 100
        self.a = alpha
        if data == 10:
            for i in range(5):
                ds = torch.utils.data.TensorDataset(set_rt[i].view(-1,3,32,32),set_y[i].view(-1))
                rl = torch.utils.data.DataLoader(ds,batch_size=100,shuffle=False)
                self.rts.append(rl)
                
        
    def SCR(self,std_train=True,std_mem=True,vir_mem=False,vir_train=False,mem_size=5000,epoch=1,num_iters=1,mem_batch=10,lr=0.1):
        Mem = Memory(mem_size=mem_size,device=self.device)
        transform = nn.Sequential(
            RandomResizedCrop(size=(32,32), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        criterion = SupConLoss(self.device)
        criterion2 = nn.CrossEntropyLoss()
        if self.data == 10:
            model = SupConResNet(num_classes=10)
        else :
            model = SupConResNet(num_classes=100)
            if self.data =='image':
                model = SupConResNet(num_classes=100,dim_in=640)
        model = model.to(self.device)
        size = self.size
        opt= torch.optim.SGD(model.parameters(),lr=lr)
        model.train()
        for i in range(len(self.set_x)):
            model.train()
            if vir_train==False:
                train_ds = D.TensorDataset(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
            else :
                train_ds = D.TensorDataset(self.set_x[i].view(-1,3,size,size),self.xv[i].view(-1,3,size,size),self.set_y[i].view(-1))
            train_loader = D.DataLoader(train_ds,batch_size=10,shuffle=True)
            if i ==0 :
                Mem.add(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
            if vir_train==False:
                for ep in range(epoch):
                    for batch_x,batch_y in  train_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        for j in range(num_iters):
                            mem_x, mem_y = Mem.pull(size=mem_batch)
                            mem_x = mem_x.to(self.device)
                            mem_y = mem_y.to(self.device)
                            combined_batch = torch.cat((mem_x,batch_x))
                            combined_labels = torch.cat((mem_y,batch_y))
                            combined_batch_aug = transform(combined_batch)
                            features = torch.cat([model(combined_batch).unsqueeze(1),model(combined_batch_aug).unsqueeze(1)],dim=1)
                            #print(features.size())
                            loss = criterion(features,combined_labels)
                            #loss = criterion2(model(batch_x),batch_y)
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                    #print(self.test(model,self.tls))
            else :
                for ep in range(epoch):
                    for batch_x,vir_x,batch_y in  train_loader:
                        batch_x = batch_x.to(self.device)
                        vir_x = vir_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        for j in range(num_iters):
                            mem_x, mem_y = Mem.pull(size=mem_batch)
                            mem_x = mem_x.to(self.device)
                            mem_y = mem_y.to(self.device)
                            aug_mem = transform(mem_x)
                            combined_batch = torch.cat((mem_x,batch_x))
                            combined_labels = torch.cat((mem_y,batch_y))
                            combined_batch_aug = torch.cat((aug_mem,vir_x))
                            features = torch.cat([model(combined_batch).unsqueeze(1),model(combined_batch_aug).unsqueeze(1)],dim=1)
                            #print(features.size())
                            loss = criterion(features,combined_labels)
                            #loss = criterion2(model(batch_x),batch_y)
                            opt.zero_grad()
                            loss.backward()
                            opt.step()                
                
            if i !=0:
                Mem.add(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
            acc = self.ncm_test(model,torch.stack(Mem.image).view(-1,3,size,size),torch.stack(Mem.label).view(-1),self.tls)
            print(np.mean(np.array(acc)))
        return np.mean(np.array(acc))
        
        
        

    def rt_test(self,model,x,y):
        ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
        ae = self.adv_attack(model,ds)
        aes = D.TensorDataset(ae,y.view(-1))
        print(self.test(model,[D.DataLoader(aes,batch_size=100,shuffle=True)]))
    
    
    def adv_training(self,model,dl,opt,epoch,test=False):
        model.train()
        losses = []
        criterion = nn.CrossEntropyLoss()
        accs = []
        adversary = LinfPGDAttack(model,self.eps,self.a,self.a_iter)
        for ep in range(epoch):
            losss = 0
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                adv = adversary.perturb(batch_x,batch_y)
                logits = model.forward(adv)
                loss = criterion(logits,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losss = loss.item()+losss
            #if ep%10 == 0 and ep!=0:
                #print(self.test(model,self.tls))
            losses.append(losss)
            if test== True:
                acc = self.test(model,self.tls)
                accs.append(acc)
        if test == True:
            return losses,accs
        return losses
    
    def training(self,model,dl,opt,epoch,test=False,show=True,full=False):
        model.train()
        losses = []
        accs = []
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            losss = 0
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                #print(batch_x.size())
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = model.forward(batch_x)
                loss = criterion(logits,batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losss = loss.item()+losss
            if ep%10 == 0 and ep!=0:
                if show==True:
                    print(self.test(model,self.tls,full=full))
            losses.append(losss)
            if test== True:
                acc = self.test(model,self.tls,full=full)
                accs.append(acc)
        if test == True:
            return losses,accs
        return losses
        
    
    def training_img(self,model,ds,m_ds,opt,epoch,test=False,show=True,mem_iter=1,mem=False,mem_batch=10):
        model.train()
        losses = []
        accs = []
        dl = D.DataLoader(ds,batch_size=100,shuffle=True)
        if mem==True:
            dl_m = D.DataLoader(m_ds,batch_size=mem_batch,shuffle=True)
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            losss = 0
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                #print(batch_x.size())
                batch_x = batch_x.view(-1,3,84,84)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if mem==True:
                    for j, mem_data in enumerate(dl_m):
                        if j < mem_iter :
                            logits = model.forward(batch_x)
                            loss = criterion(logits,batch_y)
                            opt.zero_grad()
                            loss.backward()
                            mem_x, mem_y = mem_data
                            mem_x = mem_x.view(-1,3,84,84).to(self.device)
                            #print(np.unique(np.array(mem_y)))
                            mem_y = mem_y.view(-1).to(self.device)
                            mem_logits = model.forward(mem_x)
                            mem_loss = criterion(mem_logits,mem_y)
                            mem_loss.backward()
                            
                else :
                        logits = model.forward(batch_x)
                        loss = criterion(logits,batch_y)
                        opt.zero_grad()
                        loss.backward()
                opt.step()
            if ep%10 == 0 and ep!=0:
                if show==True:
                    print(self.test(model,self.tls))
            if test== True:
                acc = self.test(model,self.tls)
                accs.append(acc)
        if test == True:
            return losses,accs
        return losses
        
    def training_cifar(self,model,ds,Mem,opt,epoch,test=False,show=True,mem_iter=1,mem=False,mem_batch=10, adv_t= False,mir=False,er=True):
        model.train()
        losses = []
        accs = []
        dl = D.DataLoader(ds,batch_size=10,shuffle=True)
        adversary = LinfPGDAttack(model,self.eps,self.a,self.a_iter)
        if self.data != 'miniimagenet':
            size=32
        else :
            size=84
        criterion = nn.CrossEntropyLoss()
        for ep in range(epoch):
            losss = 0
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                batch_x = batch_x.view(-1,3,size,size)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.view(-1).to(self.device)
                
                if mem==True:
                    if er==False:
                        for j in range(mem_iter):
                                logits = model.forward(batch_x)
                                loss = criterion(logits,batch_y)
                                losss += loss.detach().to('cpu')
                                opt.zero_grad()
                                loss.backward()
                                if adv_t == True:
                                    adv = adversary.perturb(batch_x,batch_y)
                                    adv_logits = model.forward(adv)
                                    adv_loss = criterion(logits,batch_y)
                                    adv_loss.backward()
                                if mir==True:
                                    mem_x, mem_y = Mem.MIR_retrieve(model,eps_mem_batch=mem_batch)
                                else : 
                                    mem_x, mem_y = Mem.pull(mem_batch)
                                mem_x = mem_x.view(-1,3,size,size).to(self.device)
                                mem_y = mem_y.view(-1).to(self.device)
                                mem_logits = model.forward(mem_x)
                                mem_loss = criterion(mem_logits,mem_y)
                                mem_loss.backward()
                                losss += mem_loss.detach().to('cpu')

                    else :
                        for j in range(mem_iter):
                                opt.zero_grad()
                                if mir==True:
                                    mem_x, mem_y = Mem.MIR_retrieve(model,eps_mem_batch=mem_batch)
                                else : 
                                    mem_x, mem_y = Mem.pull(mem_batch)
                                mem_x = mem_x.view(-1,3,size,size).to(self.device)
                                #print(np.unique(np.array(mem_y)))
                                mem_y = mem_y.view(-1).to(self.device)
                                inputs = torch.cat([batch_x,mem_x])
                                labels = torch.cat([batch_y,mem_y])
                                logits = model(inputs)
                                #print(mem_y.shape,mem_logits.shape)
                                mem_loss = criterion(logits,labels)
                                mem_loss.backward()  
                                losss += mem_loss.detach().to('cpu')
                else :
                    logits = model.forward(batch_x)
                    loss = criterion(logits,batch_y)
                    losss += loss.detach().to('cpu')
                    opt.zero_grad()
                    loss.backward()
                opt.step()
            if test== True:
                acc = self.test(model,self.tls)
                #print(acc)
                accs.append(acc)
            losses.append(losss)
        if test == True:
            return np.array(losses),np.array(accs)
        return np.array(losses)
    
    def review_trick(self,model,Mem,opt):
        x = torch.stack(Mem.image)
        y = torch.stack(Mem.label)
        ds = D.TensorDataset(x.view(-1,3,self.size,self.size),y.view(-1))
        dl = D.DataLoader(ds,batch_size=10,shuffle=True)
        criterion = nn.CrossEntropyLoss()
        for xs,ys in dl:
            xs = xs.to(self.device).view(-1,3,self.size,self.size)
            ys = ys.to(self.device).view(-1)
            #print(xs.size(),ys.size())
            feature = model(xs)
            loss = criterion(feature,ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    def training_womem(self,model,ds,ds_m,opt,epoch,test=False,show=True,mem_iter=1,mem=False,mem_batch=10,adv_t=False,mir=True):
        model.train()
        losses = []
        accs = []
        adversary = LinfPGDAttack(model,self.eps,self.a,self.a_iter)
        dl = D.DataLoader(ds,batch_size=10,shuffle=True)
        if mir==True:
            Mem = Memory(5000,device=self.device,subsample= 50 )
        if self.data != 'miniimagenet':
            size=32
        else :
            size=84
        criterion = nn.CrossEntropyLoss()
        if mem == True:
            dl_m = D.DataLoader(ds_m,batch_size=10,shuffle=True)
        for ep in range(epoch):
            losss = 0
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                #print(batch_x.size())
                batch_x = batch_x.view(-1,3,size,size)
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.view(-1).to(self.device)
                if mem==True:
                    if mir==False:
                        dl_m = D.DataLoader(ds_m,batch_size=mem_batch,shuffle=True)
                        for j,(mem_x,mem_y) in enumerate(dl_m):
                            if j<mem_iter:
                                logits = model.forward(batch_x)
                                loss = criterion(logits,batch_y)
                                opt.zero_grad()
                                loss.backward()
                                mem_x = mem_x.view(-1,3,size,size).to(self.device)
                                mem_y = mem_y.view(-1).to(self.device)
                                mem_logits = model.forward(mem_x)
                                mem_loss = criterion(mem_logits,mem_y)
                                mem_loss.backward()
                    if mir == True:
                        dl_m = D.DataLoader(ds_m,batch_size=100,shuffle=True)
                        for j,(mem_x,mem_y) in enumerate(dl_m):
                            if j<mem_iter:
                                logits = model.forward(batch_x)
                                loss = criterion(logits,batch_y)
                                opt.zero_grad()
                                loss.backward()
                                mem_x,mem_y = Mem.MIR_retrieve(model,input_batch = (mem_x.view(-1,3,size,size),mem_y.view(-1)))
                                mem_x = mem_x.view(-1,3,size,size).to(self.device)
                                mem_y = mem_y.view(-1).to(self.device)
                                mem_logits = model.forward(mem_x)
                                mem_loss = criterion(mem_logits,mem_y)
                                mem_loss.backward()
                else :
                        logits = model.forward(batch_x)
                        loss = criterion(logits,batch_y)
                        opt.zero_grad()
                        loss.backward()
                if adv_t == True:
                    adv = adversary.perturb(batch_x,batch_y)
                    adv_logits = model.forward(adv)
                    adv_loss = criterion(adv_logits,batch_y)
                    adv_loss.backward()                   
                opt.step()
            if ep%10 == 0 and ep!=0:
                if show==True:
                    print(self.test(model,self.tls))
            if test== True:
                acc = self.test(model,self.tls)
                accs.append(acc)
        if test == True:
            return losses,accs
        return losses
        
        
        
                
        
    def adv_attack(self,model,ds,a_iter=7):
        model.eval()
        adversary = LinfPGDAttack(model,self.eps,self.a,a_iter)
        dl = D.DataLoader(ds,batch_size=100)
        ae = []
        with torch.no_grad():
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                adv = adversary.perturb(batch_x,batch_y).to('cpu')
                ae.append(adv)
        aes = torch.stack(ae).view(-1,3,self.size,self.size)
        #print(aes.shape)
        return aes
    
    def adv_attack_img(self,model,ds,a_iter=7):
        model.eval()
        adversary = LinfPGDAttack(model,self.eps,self.a,a_iter)
        dl = D.DataLoader(ds,batch_size=1)
        ae = []
        with torch.no_grad():
            for i, batch_data in enumerate(dl):
                batch_x,batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                adv = adversary.perturb(batch_x,batch_y).to('cpu')
                ae.append(adv)
        aes = torch.stack(ae)
        print(aes.shape)
        return aes
            
    def rt_test(self,model,rts):
        accs = []
        model.eval()
        for tl in rts:
            correct = 0
            total = 0
            for x,y in tl:
                x = x.to(self.device)
                y = y.to(self.device)
                total += y.size(0)
                output = model(x)
                _,predicted = output.max(1)
                correct += predicted.eq(y).sum().item()
            accs.append(100*correct/total)
        model.train()
        return accs
            
            
    
    def test(self,model,tls,full=False):
        accs = []
        model.eval()
        if full==False:
            for tl in tls:
                correct = 0
                total = 0
                for x,y in tl:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    total += y.size(0)
                    output = model(x)
                    _,predicted = output.max(1)
                    correct += predicted.eq(y).sum().item()
                accs.append(100*correct/total)
        else :
            correct = 0
            total = 0
            for tl in tls:
                for x,y in tl:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    total += y.size(0)
                    output = model(x)
                    _,predicted = output.max(1)
                    correct += predicted.eq(y).sum().item()
            accs.append(100*correct/total)
        model.train()
        return accs

    
    def ncm_test(self,model,mem_x,mem_y,tls):
        classes = []
        model.eval()
        labels = np.unique(np.array(mem_y))
        classes=labels
        exemplar_means = {}
        size = self.size
        #print(size)
        cls_sample = {label : [] for label in labels}
        ds = D.TensorDataset(mem_x.view(-1,3,size,size),mem_y.view(-1))
        dl = D.DataLoader(ds,batch_size=1,shuffle=False)
        accs = []
        for image,label in dl:
            cls_sample[label.item()].append(image)
        for cl, exemplar in cls_sample.items():
            features = []
            for ex in exemplar:
                feature = model.features(ex.view(-1,3,size,size).to(self.device)).detach().clone()
                #print(feature.size())
                feature.data= feature.data/feature.data.norm()
                features.append(feature)
                
            if len(features)==0:
                mu_y = torch.normal(0,1,size=tuple(model.features(x.view(-1,3,size,size)).detach().size()))
            else :
                features = torch.stack(features)
                mu_y = features.mean(0)
            mu_y.data = mu_y.data/mu_y.data.norm()
            exemplar_means[cl] = mu_y
        with torch.no_grad():
            model = model
            for task, test_loader in enumerate(tls):
                acc = []
                correct = 0
                size =0
                for  batch_x,batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    feature = model.features(batch_x)
                    feature.data = feature.data / feature.data.norm()
                    if self.size == 84:
                        #print('!')
                        feature = feature.view(-1,640,1)
                        means = torch.stack([exemplar_means[cl] for cl in classes]).view(-1,640)
                    else :
                        #print('!!')
                        feature = feature.view(-1,160,1)
                        means = torch.stack([exemplar_means[cl] for cl in classes]).view(-1,160)
                    #print(feature.size())
                    #print([exemplar_means[cl] for cl in classes])
                    
                    means = torch.stack([means] * batch_x.size(0))
                    means =  means.transpose(1,2)
                    feature = feature.expand_as(means)
                    dists = (feature-means).pow(2).sum(1).squeeze()
                    _,pred_label = dists.min(1)
                    correct_cnt = (np.array(classes)[pred_label.tolist()]==batch_y.cpu().numpy()).sum().item()
                    correct += correct_cnt
                    size += batch_y.size(0)
                accs.append(correct/size)
            return accs
                

    
    def make_eae(self,iteration=7,lr=0.1,epoch=15,test=False,reduced=True):
        size = 32
        aes = []
        self.a_iter = iteration
        for i in range(len(self.set_x)):
            print(i)
            if self.data != 10:
                model = reduced_ResNet18(100)
                if self.data == 'image':
                    model.linear = torch.nn.Linear(640,100)
                    size= 84
            else : 
                if reduced==False:
                    model = ResNet18(10)
            model = model.to(self.device)
            optimizer = optim.SGD(model.parameters(),lr=lr)
            ds = D.TensorDataset(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            if self.data =='image':
                self.adv_training(model,dl,optimizer,epoch)
            else :
                self.adv_training(model,dl,optimizer,epoch)
            print(self.test(model,self.tls))
            ae = self.adv_attack(model,ds)
            if test==True:
                model_ = reduced_ResNet18(100)
                if self.data == 'image':
                    model_.linear = torch.nn.Linear(640,100)
                model_.to(self.device)
                optimizer_ = optim.SGD(model_.parameters(),lr=lr)
                ds_ = D.TensorDataset(ae.view(-1,3,size,size),self.set_y[i].view(-1))
                dl_ = D.DataLoader(ds,batch_size=100,shuffle=True)
                self.training(model_,dl_,optimizer_,5)
                print('test :',self.test(model_,self.tls))
            aes.append(ae)
        return aes


    def createDirectory(self,directory): 
        try: 
            if not os.path.exists(directory):
                os.makedirs(directory) 
        except OSError:
            print("Error: Failed to create the directory.")
         
        
        
    def replay(self,std_train=False,vir_train=False,std_mem = False,vir_mem=False,mem_size=5000, lr=0.1, epoch=1,mem_iter=1,batch_size=10,mem_batch=10,ncm=False,mir=False,add_m=False,rv=False,iteration=False,subsample=50,show=False,instant = False,reduced=True,er=False,eae_epoch=1,test=False,epoch_control=False,save=False):
        name = 'mem_size_'+str(mem_size)
        if mir==True:
            name += '+MIR'
        else :
            name += '+ER'
        if rv == True:
            name += '+RV'
        if add_m == True:
            name += '+non_random'
        if vir_train == True:
            name += '+EAT'
        if iteration == False:
            print(name)
        size= self.size
        Mem = Memory(mem_size,size=size,subsample=subsample,device= self.device)
        if self.data != 10:
            model = reduced_ResNet18(100)
        else:
            model = reduced_ResNet18(10)
        if reduced == False:
            if self.data != 10:
                model = ResNet18(100)
            else :
                model = ResNet18(10)
        
        if self.data == 'miniimagenet':
            model.linear=torch.nn.Linear(640,100)
        model = model.to(self.device)
        optimizer = optim.SGD(model.parameters(),lr=lr)
        accs = []
        t_accs = []
        losses = []
        for i in range(0,len(self.set_x)):
            x = []
            y = []
            if std_train==True:
                x.append(self.set_x[i].view(-1,3,size,size))
                y.append(self.set_y[i].view(-1))
            if vir_train==True:
                if instant == False:
                    x.append(self.xv[i].view(-1,3,size,size))
                    y.append(self.set_y[i].view(-1))
                else :
                    if self.data != 10:
                        model_ = reduced_ResNet18(100)
                    else:
                        model_ = reduced_ResNet18(10)

                    if self.data == 'miniimagenet':
                        model_.linear=torch.nn.Linear(640,100)
                    model_ = model_.to(self.device)
                    optimizer_ = optim.SGD(model_.parameters(),lr=lr)
                    ds_ = D.TensorDataset(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
                    dl_ = D.DataLoader(ds_,batch_size=100,shuffle=True)
                    self.adv_training(model_,dl_,optimizer_,eae_epoch)
                    ae = self.adv_attack(model_,ds_)
                    x.append(ae.view(-1,3,size,size))
                    y.append(self.set_y[i].view(-1))
            x = torch.cat(x)
            y = torch.cat(y)
            training_data = D.TensorDataset(x.view(-1,3,size,size),y.view(-1))
            if epoch_control !=False:
                epoch = epoch_control[i]
            if i !=0:                
                if test == True:
                    loss,t_acc = self.training_cifar(model,training_data,Mem,optimizer,epoch,mem=True,mem_batch=mem_batch,mir=mir,er=er,test=True)
                    losses.append(loss)
                    t_accs.append(t_acc)
                else :
                    loss = self.training_cifar(model,training_data,Mem,optimizer,epoch,mem=True,mem_batch=mem_batch,mir=mir,er=er)
                    losses.append(loss)
            else :
                if test == True:
                    loss,t_acc = self.training_cifar(model,training_data,[],optimizer,epoch,mem=False,mem_batch=mem_batch,test=True)
                    losses.append(loss)
                    t_accs.append(t_acc)
                else:
                    loss = self.training_cifar(model,training_data,[],optimizer,epoch,mem=False,mem_batch=mem_batch)
                    losses.append(loss)
            if std_mem == True:
                if add_m == True:
                    Mem.add_m(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
                else :
                    Mem.add(self.set_x[i].view(-1,3,size,size),self.set_y[i].view(-1))
                if self.transform==True:
                    Mem.add(self.set_x_na[i].view(-1,3,size,size),self.set_y_na[i].view(-1))
            if vir_mem == True:
                Mem.add(self.xv[i].view(-1,3,size,size),self.set_y[i].view(-1))
            #print(torch.stack(Mem.image).size())
            if rv == True and i!=0:
                optimizer_m = optim.SGD(model.parameters(),lr=lr*0.1)
                self.review_trick(model,Mem,optimizer_m)
            acc = self.test(model,self.tls)
            accs.append(np.array(acc))
            #print(acc)
            if show==True:
                print(acc)
            if ncm == True:
                ncm_acc = self.ncm_test(model,torch.stack(Mem.image).view(-1,3,size,size),torch.stack(Mem.label).view(-1),self.tls)
        if iteration ==False:
            print(acc)
            print('acc : ',np.mean(np.array(acc)))
        acc_df = []
        print('t_accs')
        for a in t_accs:
            for ass in a:
                acc_df.append(ass)
        print('losses')
        for b in losses:
            print(b)
        acc_df = pd.DataFrame(np.stack(acc_df))
        loss_df = pd.DataFrame(np.stack(losses))
        acc_df.to_csv('t_acc.csv',index=False)
        loss_df.to_csv('loss.csv',index=False)
        if save == False:
            acc_df.to_csv('t_acc.csv',index=False)
            loss_df.to_csv('loss.csv',index=False)
        else:
            self.createDirectory(save)
            acc_df.to_csv(save+'/t_acc.csv',index=False)
            loss_df.to_csv(save+'/loss.csv',index=False)
        if ncm==True:
            if iteration ==False:
                print('ncm_acc :',np.mean(np.array(ncm_acc)))
            return np.mean(np.array(acc)), np.mean(np.array(ncm_acc))
        return np.mean(np.array(acc))

    

        

        
        
    def gdumb(self,mem_size=1000,reduced=True,vir=False,std=True,lr=0.1,epoch=30,image=False):
        if reduced==True:
            if image==True:
                model = reduced_ResNet18(num_classes=100)
                model.linear = torch.nn.Linear(640,100,bias=True)
                model=model.to(self.device)
            else :
                model = reduced_ResNet18(num_classes=self.data)
                model.linear = torch.nn.Linear(160,100,bias=True)
                model=model.to(self.device)
        else: 
            model = ResNet18(num_classes=self.data).to(self.device)
        optimizer = optim.SGD(model.parameters(),lr=lr)
        x_ = []
        y_ = []
        size = int(mem_size/len(self.set_x))
        if image==False:
            print(size)
            for j in range(len(self.set_x)):
                if std == True:
                    x_.append(self.set_x[j][100:100+size].view(-1,3,32,32))
                    y_.append(self.set_y[j][100:100+size].view(-1))
                if vir == True:
                    x_.append(self.xv[j][:size].view(-1,3,32,32))
                    y_.append(self.yv[j][:size].view(-1))
            x = torch.cat(x_)
            print(x.size())
            y = torch.cat(y_)
            ds = D.TensorDataset(x,y)
            dl = D.DataLoader(ds,batch_size=10,shuffle=True)
            _ = self.training(model,dl,optimizer,epoch,test=False,show=True)
            acc = self.test(model,self.tls)
            print(acc)
            print(np.mean(np.array(acc)))
        else:
            print(size)
            for j in range(len(self.set_x)):
                if std == True:
                    x_.append(self.set_x[j][100:100+size].view(-1,3,84,84))
                    y_.append(self.set_y[j][100:100+size].view(-1))
                if vir == True:
                    x_.append(self.xv[j][:size].view(-1,3,32,32))
                    y_.append(self.yv[j][:size].view(-1))
            x = torch.cat(x_)
            print(x.size())
            y = torch.cat(y_)
            ds = D.TensorDataset(x,y)
            dl = D.DataLoader(ds,batch_size=10,shuffle=True)
            _ = self.training(model,dl,optimizer,epoch,test=False,show=True)
            acc = self.test(model,self.tls)
            print(acc)
            print(np.mean(np.array(acc)))            
        


        
        
    def one_step_replay(self,f_training,memory_sel,s_training,mem_size,epoch=30,lr=0.01):
        model = ResNet18(num_classes=self.data)
        if self.data == 100:
            palette = False
        else :
            palette= True
        name = f_training+'_'+memory_sel+'_'+s_training+'_one_step'
        model = model.to(self.device)
        if name not in os.listdir():
            os.mkdir(name)
        optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
        if f_training == 'std':
            ds = D.TensorDataset(self.set_x[0].view(-1,3,32,32),self.set_y[0].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
            #loss =  T.training_s(model,dl,[0],30)
            robustness = self.rt_test(model,self.rts)
        elif f_training == 'adv':
            ds = D.TensorDataset(self.set_x[0].view(-1,3,32,32),self.set_y[0].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_1,acc_1 = self.adv_training(model,dl,optimizer,epoch,test=True)
            robustness = self.rt_test(model,self.rts)
        elif f_training == 'e_ae':
            ds = D.TensorDataset(self.xv[0].view(-1,3,32,32),self.set_y[0].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
            robustness = self.rt_test(model,self.rts)
        elif f_training == 'mix':
            x = torch.cat([self.xv[0].view(-1,3,32,32),self.set_x[0].view(-1,3,32,32)])
            y = torch.cat([self.set_y[0].view(-1),self.set_y[0].view(-1)])
            ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
        if memory_sel == 'ori':
            #print('use original memory')
            mem_x = self.set_x[0][:1000]
            mem_y = self.set_y[0][:1000]
        elif memory_sel == 'adv':
            ae = self.adv_attack(model,ds)
            mem_x = ae[:1000]
            mem_y = self.set_y[0][:1000]
        elif memory_sel == 'e_ae':
            mem_x= self.xv[0][:1000]
            mem_y = self.set_y[0][:1000]
        elif memory_sel == 'mix':
            memory.add(self.xv[0],self.set_y[0])
        elif memory_sel == 'multi':
            memory.add(self.set_x[0],self.set_y[0])
        a = Analysis(self.set_x,self.set_y,self.device)
        if memory_sel == 'mix':
            mem_x = torch.cat([self.xv[0][:1000].view(-1,3,32,32),self.set_x[0][:1000].view(-1,3,32,32)])
            mem_y = torch.cat([self.set_y[0][:1000].view(-1),self.set_y[0][:1000].view(-1)])
        fig_0 = plt.figure()
        mem = D.TensorDataset(self.set_x[0][:1000].view(-1,3,32,32),self.set_y[0][:1000].view(-1))
        cur = D.TensorDataset(self.set_x[0][1000:].view(-1,3,32,32),self.set_y[0][1000:].view(-1))
        a.tsne_new(model,[mem,cur],['mem','cur'],name+'/tsne_before',palette=['#FF0000','#0000FF','#F4FA58','#2EFEF7'])        
        if s_training == 'std':
            x = torch.cat([mem_x.view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
            y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
            ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
            loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
            robustness_2 =self.rt_test(model,self.rts)
        elif s_training == 'adv':
            x = torch.cat([mem_x.view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
            y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
            ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
            loss_2,acc_2 = self.adv_training(model,dl,optimizer,epoch,test=True)
            robustness_2 =self.rt_test(model,self.rts)
        elif s_training == 'e_ae':
            x = torch.cat([mem_x.view(-1,3,32,32),self.xv[1].view(-1,3,32,32)])
            y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
            ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
            cur = D.TensorDataset(self.xv[1].view(-1,3,32,32),self.set_y[1].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
            robustness_2 =self.rt_test(model,self.rts)
        elif s_training == 'mix':
            x = torch.cat([mem_x.view(-1,3,32,32),self.xv[1].view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
            y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1),self.set_y[1].view(-1)])
            ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
            cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
            dl = D.DataLoader(ds,batch_size=100,shuffle=True)
            loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
            robustness_2 =self.rt_test(model,self.rts)
        
        fig_1 = plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loss_1)
        plt.savefig(name+'/loss_1.jpg',dpi=500)
        fig_2 = plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loss_2)
        plt.savefig(name+'/loss_2.jpg',dpi=500)
        fig_3 = plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        time = [i for i in range(epoch)]
        acc_1 = np.array(acc_1)
        plt.plot(time,acc_1[:,0])
        plt.savefig(name+'/acc_1.jpg',dpi=500)
        fig_4 = plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        acc_2 = np.array(acc_2)
        plt.plot(time,acc_2[:,0])
        plt.plot(time,acc_2[:,1])
        plt.savefig(name +'/acc_2.jpg',dpi=500)
        mem = D.TensorDataset(self.set_x[0][:1000].view(-1,3,32,32),self.set_y[0][:1000].view(-1))
        cur = D.TensorDataset(self.set_x[0][1000:].view(-1,3,32,32),self.set_y[0][1000:].view(-1))
        a.tsne_new(model,[mem,cur],['in-memory','out-memory'],name+'/tsne',palette=['#FF0000','#0000FF','#F4FA58','#2EFEF7'])
        a.tsne_new(model,[ds],['ds'],'test',palette=False)
    
    
    def one_step_replay_iter(self,f_training,memory_sel,s_training,mem_size,ite=1,epoch=30,lr=0.01):
        accs = []
        for i in range(ite):
            model = ResNet18(num_classes=self.data)
            name = f_training+'_'+memory_sel+'_'+s_training+'_one_step'
            if self.v != 1:
                name=name+'v-'+str(self.v)
            print(name)
            model = model.to(self.device)
            if name not in os.listdir():
                os.mkdir(name)
            optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
            memory = Memory(mem_size)
            if f_training == 'std':
                ds = D.TensorDataset(self.set_x[0].view(-1,3,32,32),self.set_y[0].view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
                #loss =  T.training_s(model,dl,[0],30)
                robustness = self.rt_test(model,self.rts)
            elif f_training == 'adv':
                ds = D.TensorDataset(self.set_x[0].view(-1,3,32,32),self.set_y[0].view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_1,acc_1 = self.adv_training(model,dl,optimizer,epoch,test=True)
                robustness = self.rt_test(model,self.rts)
            elif f_training == 'e_ae':
                ds = D.TensorDataset(self.xv[0].view(-1,3,32,32),self.set_y[0].view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
                robustness = self.rt_test(model,self.rts)
            elif f_training == 'mix':
                x = torch.cat([self.xv[0].view(-1,3,32,32),self.set_x[0].view(-1,3,32,32)])
                y = torch.cat([self.set_y[0].view(-1),self.set_y[0].view(-1)])
                ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_1,acc_1 = self.training(model,dl,optimizer,epoch,test=True)
            if memory_sel == 'ori':
                memory.add(self.set_x[0][:1000],self.set_y[0][:1000])
            elif memory_sel == 'adv':
                ae = self.adv_attack(model,ds)
                memory.add(ae[:1000],self.set_y[0][:1000])
            elif memory_sel == 'e_ae':
                memory.add(self.xv[0][:1000],self.set_y[0][:1000])
            else :
                memory.add(self.xv[0],self.set_y[0])
            a = Analysis(self.set_x,self.set_y,self.device)
            mem_x,mem_y = memory.pop(1)   
            if memory_sel == 'mix':
                mem_x = torch.cat([self.xv[0][:500].view(-1,3,32,32),self.set_x[0][:500].view(-1,3,32,32)])
                mem_y = torch.cat([self.set_y[0][:500].view(-1),self.set_y[0][:500].view(-1)])
            if s_training == 'std':
                x = torch.cat([mem_x.view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
                y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
                ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
                loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
                robustness_2 =self.rt_test(model,self.rts)
            elif s_training == 'adv':
                x = torch.cat([mem_x.view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
                y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
                ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
                loss_2,acc_2 = self.adv_training(model,dl,optimizer,epoch,test=True)
                robustness_2 =self.rt_test(model,self.rts)
            elif s_training == 'e_ae':
                x = torch.cat([mem_x.view(-1,3,32,32),self.xv[1].view(-1,3,32,32)])
                y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1)])
                ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
                cur = D.TensorDataset(self.xv[1].view(-1,3,32,32),self.set_y[1].view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
                robustness_2 =self.rt_test(model,self.rts)
            elif s_training == 'mix':
                x = torch.cat([mem_x.view(-1,3,32,32),self.xv[1].view(-1,3,32,32),self.set_x[1].view(-1,3,32,32)])
                y = torch.cat([mem_y.view(-1),self.set_y[1].view(-1),self.set_y[1].view(-1)])
                ds = D.TensorDataset(x.view(-1,3,32,32),y.view(-1))
                cur = D.TensorDataset(self.set_x[1].view(-1,3,32,32),self.set_y[1].view(-1))
                dl = D.DataLoader(ds,batch_size=100,shuffle=True)
                loss_2,acc_2 = self.training(model,dl,optimizer,epoch,test=True)
                robustness_2 =self.rt_test(model,self.rts)
            accs.append(pd.DataFrame(acc_2))
        df = pd.concat(accs)
        df.to_csv(name+'/acc.csv',index=False)
