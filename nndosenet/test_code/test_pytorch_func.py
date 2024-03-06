
import torch
from torch import nn
from torch.nn import L1Loss as MAE
import numpy as np
import torch


l=torch.randint(0,3,(3,3,3,3)).to('cuda')
x=torch.rand((3,3,3,3)).to('cuda')
y=torch.rand((3,3,3,3)).to('cuda')


def get_roi_indexs(x:torch.Tensor, axes=None, square=False):

    with torch.no_grad():
        device=x.device
        x=torch.split(x,1,dim=1)
        indexs=[]
        for c in range(0,len(x)):
            temp=x[c].flatten()
            label=temp.unique()
            for l in label:
                if l ==0: 
                    continue
                where=(temp == l).nonzero().flatten().to(device)
                indexs.append(where)

    return indexs




class value_DVH_Loss(nn.Module):
    def __init__(self, ddp: bool = True):
        """
        """
        super(value_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index):
        # print(x.shape,"!!!!!!!!!!!!!!!!!!!!!!11")
        with torch.no_grad():
            x=x.flatten()
            y=y.flatten()
            maes=torch.tensor(0)
            L=len(mask_index)
            
            for l in mask_index:

                dose_x=torch.take(x,l)
                dose_y=torch.take(y,l) 
                                
                dose_x,_=torch.sort(dose_x)
                dose_y,_=torch.sort(dose_y) 
                
                maes=maes+torch.mean(torch.abs(dose_x-dose_y))




        return maes/L
indexs=get_roi_indexs(l)
print(indexs)
DVH = value_DVH_Loss(True)
dvh_loss = DVH(x, y, mask_index=indexs)
print(dvh_loss)


