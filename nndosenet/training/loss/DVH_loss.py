# from typing import Callable

import torch
# from nndosenet.utilities.ddp_allgather import AllGatherGrad
# from nndosenet.utilities.tensor_utilities import sum_tensor
from torch import nn


class value_DVH_Loss(nn.Module):
    def __init__(self, ddp: bool = True):
        """
        """
        super(value_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index,channel_ptv_oar=None):
        with torch.no_grad():
            
            mask_indexs=mask_index[0]+mask_index[1]
            maes=torch.tensor(0.0,device=x.device)           
            
            L=len(mask_indexs)
            if L<=0:
                return None
            for l in mask_indexs:

                dose_x=torch.take(x,l)
                dose_y=torch.take(y,l) 
                
                maes+=torch.mean(torch.abs(torch.sort(dose_x)[0]-torch.sort(dose_y)[0]))
            

        return maes/L

class criteria_DVH_Loss(nn.Module):
    def __init__(self, ddp: bool = True):

        super(criteria_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index,channel_ptv_oar=None):
        with torch.no_grad():
            if len(mask_index)!=len(channel_ptv_oar):
                print(" channel_ptv_oar len don't match !!!!! c-dvh return None",len(mask_index),len(channel_ptv_oar))
                return None
            
            criteria=torch.tensor(0.).to(x.device)

            L=0
            
            #PTV/Target
            if channel_ptv_oar[0]>0:
                label=mask_index[channel_ptv_oar[0]]
                if len(label)>0:
                    L+=len(label)*3
                    for l in label:

                        dose_x=torch.take(x,l)
                        dose_y=torch.take(y,l)  
                        criteria+=torch.abs(torch.quantile(dose_x,0.01)-torch.quantile(dose_y,0.01))#d99
                        criteria+=torch.abs(torch.quantile(dose_x,0.05)-torch.quantile(dose_y,0.05))#d95
                        criteria+=torch.abs(torch.quantile(dose_x,0.99)-torch.quantile(dose_y,0.99))#d01

            #OAR
            if channel_ptv_oar[1]>0:
                label=mask_index[channel_ptv_oar[1]]
                if len(label)>0:
                    L+=len(label)*2
                    for l in label:

                        dose_x=torch.take(x,l)
                        dose_y=torch.take(y,l)  

        
                        fractional_volume_to_evaluate = 1 - 1 / len(l)
                        criteria+=torch.abs(torch.quantile(dose_x, fractional_volume_to_evaluate)-torch.quantile(dose_y, fractional_volume_to_evaluate))
                        criteria+=torch.abs(torch.mean(dose_x)-torch.mean(dose_y))#mean  
            if L==0:
                return None        
               
        return criteria/L




