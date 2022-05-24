import torch
import numpy as np
import copy
import torch.nn.functional as F

class Memory():
    def __init__(self,mem_size,size=32,subsample=100,device='cpu'):
        self.mem_size = mem_size
        self.image = []
        self.label = []
        self.num_tasks = 0
        self.image_size=size
        self.device = device
        self.subsample = subsample
    
    def add(self,image,label):
        self.num_tasks +=1
        image_new= []
        label_new = []
        task_size = int(self.mem_size/self.num_tasks)
        if self.num_tasks != 1 :
            for task_number in range(len(self.label)):
                numbers = np.array([i for i in range(len(self.label[task_number]))])
                choosed = np.random.choice(numbers,task_size,replace=False)
                image_new.append(self.image[task_number][choosed])
                label_new.append(self.label[task_number][choosed])
        numbers = np.array([i for i in range(len(label))])
        choosed = np.random.choice(numbers,task_size)
        image_new.append(image[choosed])
        label_new.append(label[choosed])
        self.image = image_new
        self.label = label_new
        #print(torch.stack(label_new).size())
        
        
    def add_m(self,image,label):
        self.num_tasks +=1
        image_new = []
        label_new = []
        task_size = int(self.mem_size/self.num_tasks)
        if self.num_tasks !=1:
            for task_number in range(len(self.label)):
                image_new.append(self.image[task_number][:task_size])
                label_new.append(self.label[task_number][:task_size])
        image_new.append(image[:task_size])
        label_new.append(label[:task_size])
        self.image = image_new
        self.label=label_new
        
    def pull(self,size):
        image = torch.stack(self.image).view(-1,3,self.image_size,self.image_size)
        label = torch.stack(self.label).view(-1)
        numbers = np.array([i for i in range(len(label))])
        choosed = np.random.choice(numbers,size,replace=False)
        return image[choosed],label[choosed]
    
     
    def MIR_retrieve(self,model,eps_mem_batch=10,input_batch = False):
        subsample = self.subsample
        if input_batch ==False:
            sub_x, sub_y = self.pull(subsample)
        else :
            sub_x, sub_y = input_batch
        sub_x = sub_x.to(self.device)
        sub_y = sub_y.to(self.device)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = self.get_grad_vector(model.parameters,grad_dims)
        model_temp = self.get_future_step_parameters(model,grad_vector,grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = model(sub_x)
                logits_post = model_temp(sub_x)
                pre_loss = F.cross_entropy(logits_pre,sub_y,reduction='none')
                post_loss = F.cross_entropy(logits_post,sub_y,reduction='none')
                scores = post_loss - pre_loss
                big_ind = scores.sort(descending = True)[1][:eps_mem_batch]
            return sub_x[big_ind],sub_y[big_ind]
        else:
            return sub_x,sub_y



    
    def get_grad_vector(self,pp,grad_dims):
        grads = torch.Tensor(sum(grad_dims)).to(self.device)
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en=sum(grad_dims[:cnt+1])
                grads[beg:en].copy_(param.grad.data.view(-1))
            cnt +=1
        return grads
    
    def get_future_step_parameters(self,model,grad_vector,grad_dims,learning_rate=0.1):
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters,grad_vector,grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - learning_rate * param.grad.data
        return new_model
    
    def overwrite_grad(self,pp,new_grad,grad_dims):
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt+1])
            this_grad = new_grad[beg:en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt +=1

                
                
                
