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

    def forward(self, x, y, mask_index):
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

class value_DVH_Loss(nn.Module):
    def __init__(self, ddp: bool = True):
        """
        """
        super(value_DVH_Loss, self).__init__()

        self.ddp = ddp

    def forward(self, x, y, mask_index):
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




