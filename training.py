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
import matplotlib.pyplot as plt



class LinfPGDAttack(object):
    def __init__(self, model,epsilon,alpha,a_iter):
        self.eps = epsilon
        self.alpha = alpha
        self.a_iter = a_iter
        self.model = model

    def perturb(self, x_natural, y):
        epsilon = self.eps
        alpha = self.alpha
        k = self.a_iter
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
    def perturb_reverse(self, x_natural, y):
        epsilon = self.eps
        alpha = self.alpha
        k = self.a_iter
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = -F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def perturb_2(self, x_natural, y):
        epsilon = self.eps
        alpha = self.alpha
        k = self.a_iter
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model.second_forward(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
    def perturb_mse(self,x_natural,x_target):
        epsilon = self.eps
        alpha = self.alpha
        k = self.a_iter
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.no_grad():
                logits_ori = self.model.semi_forward(x_target)
            with torch.enable_grad():
                logits = self.model.semi_forward(x)
                loss = -F.mse_loss(logits,logits_ori)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


class Trainer(object):
    def __init__(self,set_x,set_y,test_loaders,device,lr=0.1,a_iter=7,eps=0.0314,alpha=0.00784,report_time=100):
        self.set_x = set_x
        self.set_y = set_y
        self.test_loaders = test_loaders
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.a_iter = a_iter
        self.eps = eps
        self.a = alpha
        self.rt = report_time
        
    def change_lr(self,lr):
        self.lr=lr
        
    def train(self,epoch,train_loader,net,optimizer):
        #print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        device = self.device
        adversary = LinfPGDAttack(net,self.eps,self.a,self.a_iter)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = self.criterion(adv_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #if batch_idx % 10 == 0:
                #print('\nCurrent batch:', str(batch_idx))
                #print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                #print('Current adversarial train loss:', loss.item())

        #print('\nTotal adversarial train accuarcy:', 100. * correct / total)
        #print('Total adversarial train loss:', train_loss)

    def train_c(self,epoch,train_loader,net,optimizer):
        device = self.device
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = SupConLoss(device)
        adversary = LinfPGDAttack(net,self.eps,self.a,self.a_iter)
        for batch_idx, (inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            adv = adversary.perturb(inputs,targets)
            features = torch.cat([net.forward(inputs).unsqueeze(1),net.forward(adv).unsqueeze(1)],dim=1)
            loss = criterion(features,targets)
            loss.backward()
            optimizer.step()
            
            
            
        
    def test(self,epoch,test_loader,net):
        #print('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        device = self.device
        adversary = LinfPGDAttack(net,self.eps,self.a,self.a_iter)
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)

                outputs = net(inputs)

                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()

                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(adv)

                _, predicted = adv_outputs.max(1)
                adv_correct += predicted.eq(targets).sum().item()

        #state = {
        #    'net': net.state_dict()
        #}
        return 100*benign_correct/total, 100*adv_correct/total
    
    def test_wo_adv(self,epoch,test_loader,net):
        #print('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        device = self.device
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 1
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)

                outputs = net(inputs)

                _, predicted = outputs.max(1)
                benign_correct += predicted.eq(targets).sum().item()
        #state = {
        #    'net': net.state_dict()
        #}
        return 100*benign_correct/total
    
    def test_2(self,model,tls):
        accs = []
        model.eval()
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
        model.train()
        return accs

    def adjust_learning_rate(self,optimizer, epoch):
        lr = self.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        if epoch >= 200:
            lr /=10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_wo_adv(self,epoch,train_loader,net,optimizer):
        #print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        device = self.device
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            benign_outputs = net(inputs)
            loss = self.criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return train_loss

    def train_i(self,epoch,train_loader,net,optimizer):
        #print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        device = self.device
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            benign_outputs = net.inject_forward(inputs,device)
            loss = self.criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  

    def train_d(self,epoch,train_loader,net,optimizer):
    #print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        device = self.device
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            adv_outputs = net.forward_d(inputs)
            loss = self.criterion(adv_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    def training_s(self,model,dataloader,num_test,epoch,plot=False,adjust=True,schedule =False):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0005)
        if schedule ==True:
            scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
        s_s = [[] for i in range(len(num_test))]
        loss = []
        for ep in range(epoch):
            if adjust==True:
                self.adjust_learning_rate(optimizer,ep)
            tl = self.train_wo_adv(ep,dataloader,model,optimizer)
            loss.append(tl)
            ss = []
            k=0
            for i in num_test:
                s = self.test_wo_adv(0,self.test_loaders[i],model)
                s_s[k].append(s)
                ss.append(s)
                k=k+1
            if ep%self.rt == 0 and ep !=0:                        
                print('std :',ss)
        s_s = np.array(s_s)
        if plot==True:
            time = [ i for i in range(epoch)]
            for s in s_s:
                plt.plot(time,s)
            plt.plot(time,s_s.mean(axis=0))
            plt.plot(time,loss)
            plt.show()
        return s_s,loss

    def training_s_2(self,model,dataloader,num_test,epoch,plot=False):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0002)
        s_s = [[] for i in range(len(num_test))]
        for ep in range(epoch):
            #self.adjust_learning_rate(optimizer,ep)
            self.train_wo_adv(ep,dataloader,model,optimizer)
            ss = []
            for i in num_test:
                s = self.test_wo_adv(0,self.test_loaders[i],model)
                s_s[i].append(s)
                ss.append(s)
                if ep%self.rt == 0 and ep !=0:                        
                    print('std :',ss)
        s_s = np.array(s_s)
        if plot==True:
            time = [ i for i in range(epoch)]
            for i in num_test:
                plt.plot(time,s_s[i])
            plt.plot(time,s_s.mean(axis=0))
            plt.show()
        return s
    
    def training_a(self,model,dataloader,num_test,epoch,adjust=True,plot=False):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0005)
        s_s = [[] for i in range(len(num_test))]
        loss = []
        for ep in range(epoch):
            if adjust==True:
                self.adjust_learning_rate(optimizer,ep)
            tl = self.train(ep,dataloader,model,optimizer)
            loss.append(tl)
            ss = []
            k=0
            for i in num_test:
                s = self.test_wo_adv(0,self.test_loaders[i],model)
                s_s[k].append(s)
                ss.append(s)
                k=k+1
            if ep%self.rt == 0 and ep !=0:                        
                print('std :',ss)
        s_s = np.array(s_s)
        if plot==True:
            time = [ i for i in range(epoch)]
            for s in s_s:
                plt.plot(time,s)
            plt.plot(time,s_s.mean(axis=0))
            plt.plot(time,loss)
            plt.show()
        return s_s
    
    def training_i(self,model,dataloader,num_test,epoch,adjust=True):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0002)
        for ep in range(epoch):
            if adjust==True:
                self.adjust_learning_rate(optimizer,ep)
            if ep<=30:
                self.train_wo_adv(ep,dataloader,model,optimizer)
            else:
                self.train_i(ep,dataloader,model,optimizer)
            if ep%self.rt==0:
                s=[]
                a=[]
                for i in num_test:
                    s1,a1 = self.test(0,self.test_loaders[i],model)
                    s.append(s1)
                    a.append(a1)
                print('std :',s)
                print('adv :',a)
        return s,a
    
    def training_c(self,model,dataloader,num_test,epoch,epoch_s=30,adjust=True,plot=True):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0002)
        s_s = [[] for i in range(len(num_test))]
        ep=0
        self.train_c(ep,dataloader,model,optimizer)
        for ep in range(epoch):
            ss=[]
            if adjust==True:
                self.adjust_learning_rate(optimizer,ep)
            if ep<=epoch_s:
                self.train_wo_adv(ep,dataloader,model,optimizer)
            else:
                self.train_c(ep,dataloader,model,optimizer)
            for i in num_test:
                s = self.test_wo_adv(0,self.test_loaders[i],model)
                s_s[i].append(s)
                ss.append(s)
            if ep%self.rt == 0 and ep !=0:                        
                print('std :',ss)
        s_s = np.array(s_s)
        if plot==True:
            time = [ i for i in range(epoch)]
            for i in num_test:
                plt.plot(time,s_s[i])
            plt.plot(time,s_s.mean(axis=0))
            plt.show()
        return s_s
    


    def training_d(self,model,dataloader,num_test,epoch,plot=False,adjust=True):
        device = self.device
        learning_rate = self.lr
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0002)
        s_s = [[] for i in range(len(num_test))]
        loss = []
        for ep in range(epoch):
            if adjust==True:
                self.adjust_learning_rate(optimizer,ep)
            tl = self.train_d(ep,dataloader,model,optimizer)
            loss.append(tl)
            ss = []
            for i in num_test:
                s = self.test_wo_adv(0,self.test_loaders[i],model)
                s_s[i].append(s)
                ss.append(s)
                if ep%self.rt == 0 and ep !=0:                        
                    print('std :',ss)
        s_s = np.array(s_s)
        if plot==True:
            time = [ i for i in range(epoch)]
            for i in num_test:
                plt.plot(time,s_s[i])
            plt.plot(time,s_s.mean(axis=0))
            plt.plot(time,loss)
            plt.show()
        return s_s

    def training_a_2(self,model,perm,mem,task,num_test,epoch):
        device = self.device
        model = model.to(device)
        set_x = self.set_x
        set_y = self.set_y
        learning_rate = self.lr
        adversary = LinfPGDAttack(model,self.eps,self.a,self.a_iter)
        optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay = 0.0002)
        perm = torch.randperm(len(set_x[mem]))
        mem_x = set_x[mem][perm[:1000]]
        mem_y = set_y[mem][perm[:1000]]
        tds = torch.utils.data.TensorDataset(mem_x.view(-1,3,32,32),mem_y.view(-1))
        tdl = torch.utils.data.DataLoader(tds,batch_size=100,shuffle=True)
        for ep in range(epoch):
            adv = []
            self.adjust_learning_rate(optimizer,epoch)
            for x,y in tdl:
                x = x.to(device)
                y = y.to(device)
                advs = adversary.perturb(x,y).to('cpu')
                adv.append(advs)
            adv = torch.stack(adv)
            td_x = torch.cat([set_x[task],adv.view(-1,1,3,32,32)])
            td_y = torch.cat([set_y[task],mem_y])
            td = torch.utils.data.TensorDataset(td_x.view(-1,3,32,32),td_y.view(-1))
            tl = torch.utils.data.DataLoader(td,batch_size=100,shuffle=True)
            self.train_wo_adv(epoch,tl,model,optimizer)
            if ep % self.rt==0:
                s=[]
                a=[]
                for i in num_test:
                    s1,a1 = self.test(0,self.test_loaders[i],model)
                    s.append(s1)
                    a.append(a1)
                print('std :',s)
                print('adv :',a)
        return s,a

    
    
    def measure_bonding(self,model,xs,ys):
        device = self.device
        model = model.to(device)
        tds = torch.utils.data.TensorDataset(xs.view(-1,3,32,32),ys.view(-1))
        tdl = torch.utils.data.DataLoader(tds,batch_size=1,shuffle=False)
        features = [[] for i in range(2)]
        dists = []
        with torch.no_grad():
            for x,y in tdl:
                x = x.to(device)
                y = y.to(device)
                if y < 5:
                    rep = model(x).to('cpu')
                    features[0].append(rep.numpy())
                else :
                    rep = model(x).to('cpu')
                    features[1].append(rep.numpy())
        features = np.array(features)
        for f in features:
            centor = np.mean(f,axis=0)
            dist = 0
            for fs in f:
                d = ((centor-fs)**2).mean(axis=0)
                dist = dist+d
            dist = dist/len(f)
            dists.append(np.mean(dist,axis=0))
        print(dists)
                
            
