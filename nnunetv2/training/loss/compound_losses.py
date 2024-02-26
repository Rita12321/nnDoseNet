import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.DVH_loss import value_DVH_Loss,criteria_DVH_Loss
from torch import nn
from torch.nn import MSELoss,L1Loss

class Err_and_DVH_Loss(nn.Module):
    def __init__(self,losses:list=[MSELoss],kwargs:[{}]=[{}],weights:list=[1], channel_ptv_oar:list=[1,2],dose_mask=None):
        """ #FIXME: change description
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Err_and_DVH_Loss, self).__init__()
        

        self.weight_dvh = []
        self.weight_err = []

        self.channel_ptv_oar=channel_ptv_oar
        #FIXME: add len loss weight detect
        # self.MSE = MSELoss(**MSE_kwargs)
        # self.DVH = DVH_class(**DVH_kwargs)
        
        #self.losses=losses
        # self.kwargs=kwargs
        #FIXME: not sure if list would slow down the calculation
        
        self.err_losses=[]
        self.dvh_losses=[]
        for i,loss in enumerate(losses):
            if (loss is value_DVH_Loss) or (loss is criteria_DVH_Loss):
                self.dvh_losses.append(loss(**kwargs[i]))
                self.weight_dvh.append(weights[i])
            else:
                self.err_losses.append(loss(**kwargs[i]))
                self.weight_err.append(weights[i])


        self.dose_mask=dose_mask
        
        #print(losses,weights)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor,data: torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print("net_output.shape",net_output.shape,"target.shape",target.shape,"data.shape",data.shape)
        # dose_mask_indx=None
        # dvh_mask_index=None
        device=data.device
        losses=torch.tensor(0.).to(device)
        if  len(self.dvh_losses)>0:
            
            if self.channel_ptv_oar is not None:
                ptv_oar=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar,return_vptv_oar=True)

            err_weight=0
            for i,loss in  enumerate(self.dvh_losses):
                dvh_loss=loss(net_output, target, mask_index=ptv_oar,channel_ptv_oar=self.channel_ptv_oar)
                if dvh_loss>0:
                    losses=losses+(self.weight_dvh[i]*dvh_loss).to(device)           
                else:
                    err_weight+=self.weight_dvh[i]
    
        if len(self.err_losses)>0:
            
            if self.dose_mask is not None:
                dose_mask_indx=data[:,self.dose_mask,...].flatten().nonzero()
                dose_x=torch.take(net_output,dose_mask_indx).to(device)
                dose_y=torch.take(target,dose_mask_indx).to(device) 
            else:
                dose_x=net_output
                dose_y=target
            del data, net_output, target
                
            for i,loss in enumerate(self.err_losses):
                losses=losses+((self.weight_err[i]+err_weight)*loss(dose_x, dose_y)).to(device)
                    

        
        return losses
        


def get_roi_indexs(x:torch.Tensor,return_label=False, channel_ptv_oar=[0,1],return_vptv_oar=False):
    with torch.no_grad():
        device=x.device
        #print("x_shape",x.shape )
        x=torch.split(x,1,dim=1)
        
        
        if return_label:
            labels=[]

        indexs=[]
        
        for c in channel_ptv_oar:
            index=[]#
            temp=x[c].flatten()
            label=temp.unique()
            if return_label:
                labels.append(label)
            for l in label:
                if l ==0: 
                    continue
                where=(temp == l).nonzero().flatten().to(device)
                index.append(where)#      
            indexs.append(index)
                
        if return_label:
            return indexs,labels
        return indexs

