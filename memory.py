import torch
import numpy as np

class Memory(object):
    def __init__(self,buffer_size):
        self.size=buffer_size
        self.img = torch.FloatTensor(buffer_size,3,32,32).fill_(0)
        self.label = torch.LongTensor(buffer_size).fill_(0)
        self.current_index= 0 
        self.n_seen_so_far = 0
        self.x = []
        self.y = []
    
    def add(self,x,y):
        self.x.append(x.view(-1,3,32,32))
        self.y.append(y.view(-1))
    
    def pop(self,i,rand=False):
        num = int(self.size/len(self.x))
        xs = []
        ys = []
        nums = []
        #print(len(self.x))
        for xss in self.x:
            nums.append(torch.randperm(len(xss)))
        if rand == False:
            for xss in self.x:            
                xs.append(xss[:num])
            for yss in self.y:
                ys.append(yss[:num])
        else:
            for i in range(len(self.x)):
                xss = self.x[i]
                yss = self.y[i]
                n = torch.randperm(len(xss))
                xs.append(xss[n[:num]])
                ys.append(yss[n[:num]])            
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs,ys
    
    def store(self,x,y):
        batch_size = x.size(0)
        place_left = max(0,self.size-self.current_index)
        if place_left :
            offset = min(place_left,batch_size)
            self.img[self.current_index:self.current_index+offset].data.copy_(x[:offset])
            self.label[self.current_index:self.current_index+offset].data.copy_(y[:offset])
            self.current_index += offset
            self.n_seen_so_far += offset
            
            if offset == x.size(0):
                filled_idx = list(range(self.current_index-offset, self.current_index,))
        x,y = x[place_left:],y[place_left:]
        indices = torch.FloatTensor(x.size(0)).uniform_(0,self.n_seen_so_far).long()
        valid_indices = (indices < self.img.size(0)).long()
        
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]
            
        self.n_seen_so_far += x.size(0)
        
        if idx_buffer.numel() == 0:
            return []
        
        assert idx_buffer.max() < self.img.size(0)
        assert idx_buffer.max() < self.label.size(0)
        # assert idx_buffer.max() < self.buffer_task.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)
        
        idx_map = {idx_buffer[i].item():idx_new_data[i].item() for i in range(idx_buffer.size(0))}

        replace_y = y[list(idx_map.values())]
        self.img[list(idx_map.keys())] = x[list(idx_map.values())]
        self.label[list(idx_map.keys())] = replace_y
        return list(idx_map.keys())
        
        
        
        
    def retrieve(self,num_retrieve,excl_indices=None,return_indices = False):
        filled_indices = np.arange(self.current_index)
        if excl_indices is not None:
            excl_indices = list(excl_indices)
        else:
            excl_indices = []
        valid_indices = np.setdiff1d(filled_indices,np.array(excl_indices))
        num_retrieve = min(num_retrieve,valid_indices.shape[0])
        indices = torch.from_numpy(np.random.choice(valid_indices,num_retrieve,replace=False)).long()
        
        x = self.img[indices]
        y = self.label[indices]
        
        if return_indices:
            return x,y, indices
        else:
            return x,y
        
        
    