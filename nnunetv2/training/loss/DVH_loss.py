from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.tensor_utilities import sum_tensor
from torch import nn


class value_DVH_Loss(nn.Module):
    def __init__(self, ddp: bool = True):
        """
        """
        super(value_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index,channel_ptv_oar=None):
        #FIXME: error with value_DvH+critera_DVH
        with torch.no_grad():
            x=x.flatten()
            y=y.flatten()
            maes=torch.tensor(0.)
            L=len(mask_index)
            
            for l in mask_index:

                dose_x=torch.take(x,l)
                dose_y=torch.take(y,l) 
                                
                dose_x,_=torch.sort(dose_x)
                dose_y,_=torch.sort(dose_y) 
                
                maes=maes+torch.mean(torch.abs(dose_x-dose_y))

        return maes/L

class criteria_DVH_Loss(nn.Module):#FIXME: haven't test yet
    def __init__(self, ddp: bool = True):

        super(criteria_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index,channel_ptv_oar=None):
        with torch.no_grad():
            if channel_ptv_oar==None:
                print("there is no channel_ptv_oar !!!!! c-dvh return 0")
                return 0
            if len(mask_index)!=len(channel_ptv_oar):
                print(" channel_ptv_oar len don't match !!!!! c-dvh return 0")
                return 0
            
            criteria=torch.tensor(0.).to(x.device)
            x=x.flatten()
            y=y.flatten()
            L=0
            
            #PTV
            c_ptv=channel_ptv_oar[0]
            label=mask_index[c_ptv]
            L+=len(label)*3
            for l in label:

                dose_x=torch.take(x,l)
                dose_y=torch.take(y,l)  
                criteria+=torch.abs(torch.quantile(dose_x,0.01)-torch.quantile(dose_y,0.01))#d99
                criteria+=torch.abs(torch.quantile(dose_x,0.05)-torch.quantile(dose_y,0.05))#d95
                criteria+=torch.abs(torch.quantile(dose_x,0.99)-torch.quantile(dose_y,0.99))#d01

            #OAR
            c_oar=channel_ptv_oar[1]
            label=mask_index[c_oar]
            L+=len(label)*2
            for l in label:

                dose_x=torch.take(x,l)
                dose_y=torch.take(y,l)  


                criteria+=torch.abs(torch.max(dose_x)-torch.max(dose_y))#cc01
                criteria+=torch.abs(torch.mean(dose_x)-torch.mean(dose_y))#mean              
               
#FIXME: L might equal 0 choose what to return , <---nan in loss
        return criteria/L




