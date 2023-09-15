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
        maes=torch.tensor(0)
        L=len(mask_index)
        for l in range(0,L):
            with torch.no_grad():
                dose_x=torch.Tensor([])
                dose_y=torch.Tensor([])

                for i  in mask_index[l]:
                    dose_x=torch.cat((dose_x,torch.Tensor([x[1,i[0],i[1],i[2]]])),0)
                    dose_y=torch.cat((dose_y,torch.Tensor([y[1,i[0],i[1],i[2]]])),0)
                    
                dose_x,_=torch.sort(dose_x)
                dose_y,_=torch.sort(dose_y) 
            maes=maes+torch.mean(torch.abs(dose_x-dose_y))
            
        # if self.ddp and self.batch_dice: #3dnet bath_dice false, so basically wont use this
        #     tp = AllGatherGrad.apply(tp).sum(0)
        #     fp = AllGatherGrad.apply(fp).sum(0)
        #     fn = AllGatherGrad.apply(fn).sum(0)



        # print(maes)

        return maes/L






if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
