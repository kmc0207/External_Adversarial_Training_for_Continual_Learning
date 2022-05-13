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
from training import Trainer
from model import ResNet18
from data import setting_data,make_test_loaders
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns


class Analysis(object):
    def __init__(self,set_x,set_y,device):
        self.device = device
        self.dic  = {0:'airplane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck',10:'mem_airplane',11:'mem_car',12:'mem_bird',13:'mem_cat',14:'mem_deer',15:'mem_dog',16:'mem_frog',17:'mem_horse',18:'mem_ship',19:'mem_truck'}
        
    def make_sample_tsne_f(self,net_ss,train_loader1_1,tags):
        i=0
        outs_s =[]
        out_y = []
        out_t = []
        with torch.no_grad():
            net_ss =net_ss.to(self.device)
            for x,y in train_loader1_1:
                x = x.to(self.device)
                if i<=1000000:
                    out = net_ss.semi_forward(x).to('cpu')
                    outs_s.append(out)
                    out_y.append('label_'+str(y.item())+'_'+tags)
                    out_t.append(tags)
                i=i+1
            outs_s = torch.stack(outs_s).view(-1,512)
            print(outs_s.shape,tags)
        return outs_s,out_y,out_t
    
    def tsne_new(self,model,dss,names,name='test',palette=False,no_label=False):
        dls = []
        model.eval()
        for ds in dss :
            dls.append(torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False))
        tfs = []
        tsne = TSNE(n_components=2,perplexity=10,n_iter=300)
        dps = []
        tfs = []
        labels = []
        tags = []
        for i in range(len(dls)):
            t,l,tag = self.make_sample_tsne_f(model,dls[i],names[i])
            tfs.append(t)
            labels.append(l)
            tags.append(tag)
        tf = np.concatenate(tfs)
        labels = np.concatenate(labels)
        tags = np.concatenate(tags)
        print(tf.shape)
        tsne_ref = tsne.fit_transform(tf)
        df = pd.DataFrame(tsne_ref,index=tsne_ref[0:,1])
        df['x'] = tsne_ref[:,0]
        df['y'] = tsne_ref[:,1]
        df['label']=labels
        df['tag']=tags
        dps.append(df)
        dfs = pd.concat(dps)
        fig = plt.figure()
        if no_label == True:
            sns.scatterplot(x='x',y='y',data=dfs,hue='label',palette=palette,s=3)
        else:
            if palette != False:
                sns.scatterplot(x='x',y='y',data=dfs,hue='label',palette=palette)
            else : 
                sns.scatterplot(x='x',y='y',data = dfs,hue='label',s=3)
        lg =plt.legend(bbox_to_anchor=(1.01,1),loc='upper left')
        plt.savefig(str(name)+'.jpg',dpi=500,bbox_inches='tight',bbox_extra_artists = (lg,))
        plt.show()
        plt.close()
        
            





    def tsne_2(self,perm,i,j,model,as_s,ss_s,types):
        dic = {0:'airplane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck',10:'mem_airplane',11:'mem_car',12:'mem_bird',13:'mem_cat',14:'mem_deer',15:'mem_dog',16:'mem_frog',17:'mem_horse',18:'mem_ship',19:'mem_truck'}
        mem_y=[]
        set_x = self.set_x
        set_y = self.set_y
        for k in perm[:1000]:
            mem_y.append(set_y[i][k]+10)
        mem_y = torch.stack(mem_y)
        x_ = torch.cat([set_x[i][perm[1000:]],set_x[j],set_x[i][perm[:1000]]])
        y_ = torch.cat([set_y[i][perm[1000:]],set_y[j],mem_y])
        td = torch.utils.data.TensorDataset(x_.view(-1,3,32,32),y_.view(-1))
        tl = torch.utils.data.DataLoader(td,batch_size=1,shuffle=False)
        test_features = self.make_sample_tsne_f(model,tl)
        tsne = TSNE(n_components=2,perplexity=10,n_iter=300)
        tf = []
        for f in test_features:
            tf.append(f[0])
        tsne_ref = tsne.fit_transform(tf)
        labels = []
        typess =[]
        for y in y_:
            labels.append(dic[y.item()])
            if y<10:
                typess.append('train')
            else : 
                typess.append('mem')
        df = pd.DataFrame(tsne_ref,index=tsne_ref[0:,1])
        df['x']=tsne_ref[:,0]
        df['y']=tsne_ref[:,1]
        df['label']=labels
        df['type']= typess
        sns.scatterplot(x='x',y='y',data=df,hue='label',s=5,style='type')
        name = str(i)+' to '+str(j)+'_'+types+'trial_'+str(num)
        title = str(as_s)+'/'+str(ss_s)
        plt.title(title)
        print('save ',name)
        plt.savefig(name+'.jpg',dpi=300)
        plt.close()

    def tsne(perm,i,j,model,types):
        dic = {0:'airplane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck',10:'mem_airplane',11:'mem_car',12:'mem_bird',13:'mem_cat',14:'mem_deer',15:'mem_dog',16:'mem_frog',17:'mem_horse',18:'mem_ship',19:'mem_truck'}
        mem_y=[]
        set_x = self.set_x
        set_y = self.set_y
        for k in perm[:1000]:
            mem_y.append(set_y[i][k]+10)
        mem_y = torch.stack(mem_y)
        x_ = torch.cat([set_x[i][perm[1000:]],set_x[i][perm[:1000]]])
        y_ = torch.cat([set_y[i][perm[1000:]],mem_y])
        td = torch.utils.data.TensorDataset(x_.view(-1,3,32,32),y_.view(-1))
        tl = torch.utils.data.DataLoader(td,batch_size=1,shuffle=False)
        test_features = make_sample_tsne_f(model,tl)
        tsne = TSNE(n_components=2,perplexity=10,n_iter=300)
        tf = []
        for f in test_features:
            tf.append(f[0])
        tsne_ref = tsne.fit_transform(tf)
        labels = []
        typess =[]
        for y in y_:
            labels.append(dic[y.item()])
            if y<10:
                typess.append('train')
            elif y>20:
                typess.apppend('adv')
            else : 
                typess.append('mem')
        df = pd.DataFrame(tsne_ref,index=tsne_ref[0:,1])
        df['x']=tsne_ref[:,0]
        df['y']=tsne_ref[:,1]
        df['label']=labels
        df['type']= typess
        sns.scatterplot(x='x',y='y',data=df,hue='label',s=5,style='type')
        name = str(i)+' will go to '+str(j)+types+'_trial_'+str(num)
        title = types+'_'+str(i)
        plt.title(title)
        print('save ',name)
        plt.savefig(name+'.jpg',dpi=300)
        plt.close()