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

    def forward(self, x:torch.Tensor, y:torch.Tensor, mask_index):
        # print(x.shape,"!!!!!!!!!!!!!!!!!!!!!!11")
        with torch.no_grad():
            # print(x.is_cuda,"net_output!!!!!!!!!!!!!!!!!!!!!!!!!!111111")
            # print(y.is_cuda,"target!!!!!!!!!!!!!!!!!!!!!!!!!!111111")


            maes=torch.tensor(0).cuda()
            L=len(mask_index)
            if L>0:
                
                for l in range(0,L):
                    
                    dose_x=torch.Tensor([]).cuda()
                    dose_y=torch.Tensor([]).cuda()

                    for i  in mask_index[l]:
                        index=(i[0],0,i[2],i[3],i[4])

                        dose_x=torch.cat((dose_x,torch.Tensor([x[index]]).cuda()),0)
                        dose_y=torch.cat((dose_y,torch.Tensor([y[index]]).cuda()),0)
                        
                    dose_x,_=torch.sort(dose_x)
                    dose_y,_=torch.sort(dose_y) 
                    maes=maes+torch.mean(torch.abs(dose_x-dose_y))
                    
                    # if self.ddp and self.batch_dice: #3dnet batch_dice false, so basically wont use this
                    #     tp = AllGatherGrad.apply(tp).sum(0)
                    #     fp = AllGatherGrad.apply(fp).sum(0)
                    #     fn = AllGatherGrad.apply(fn).sum(0)
                    #print(maes.is_cuda,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")

                    # print(maes)
            else:
                L=1# prevent from /0-->nan

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
