from __future__ import division #NOTE: this prevent errors warning GPU0 <> GPU1 (GPU1 has < 75% capacity of GPU0)
import numpy as np
import pickle
import cv2
import imp

import torch as t
import torchvision as tv
import torchvision.transforms as transform
#from torch.utils.data import Dataset, DataLoader

import header_headoren_salcorr as header
import header_saliency_ds as header_sal


# H_SAL = 90
# W_SAL = 160

def modify_fc_layers(model):
    model.avgpool = t.nn.AdaptiveAvgPool2d(3)
    model.fc = t.nn.Linear(3 * 3 * 2048, header_sal.TARGET_SAL_H*header_sal.TARGET_SAL_W) # assuming that the fc7 layer has 512 neurons, otherwise change it 
    
    
class PanoSalDataset(t.utils.data.Dataset):
    def __init__(self, pickle_file, transform=None):
        self._pickle_file = pickle_file
        self._dat = pickle.load(open(pickle_file, 'rb'))
        self.transform = transform
    
    def __len__(self):
        return len(self._dat)
    
    def __getitem__(self, idx):
        if t.is_tensor(idx):
            idx = idx.tolist()
        t0, img, smap = self._dat[idx]
        img = self.transform(img)
        #print (f'Before: {img.shape}')
        #img = img.transpose(2, 0, 1)
        #print (f'After: {img.shape}')
        return t0, img, smap
    
def get_lr(optimizer):#directly change learning rate of the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']
        