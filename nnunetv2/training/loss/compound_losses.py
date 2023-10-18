import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.DVH_loss import value_DVH_Loss,criteria_DVH_Loss
from torch import nn
from torch.nn import MSELoss,L1Loss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target must be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

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
        
        losses=torch.tensor(0.)
        if  len(self.dvh_losses)>0:
            
            if self.channel_ptv_oar is not None:
                ptv_oar=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar,return_vptv_oar=True)
            
            for i,loss in  enumerate(self.dvh_losses):
                #if self.weight_dvh[i]!=0:
                losses=losses+self.weight_dvh[i]*loss(net_output, target, mask_index=ptv_oar,channel_ptv_oar=self.channel_ptv_oar) 
                #losses=losses+loss(net_output, target, mask_index=ptv_oar,channel_ptv_oar=self.channel_ptv_oar)             
    
        if len(self.err_losses)>0:

            if self.dose_mask is not None:
                dose_mask_indx=data[self.dose_mask].flatten().nonzero()
                dose_x=torch.take(net_output,dose_mask_indx)
                dose_y=torch.take(target,dose_mask_indx) 
            else:
                dose_x=net_output
                dose_y=target
            del data, net_output, target
                
            for i,loss in enumerate(self.err_losses):
                #if self.weight_err[i]!=0:
                losses=losses+self.weight_err[i]*loss(dose_x, dose_y)
                #losses=losses+loss(dose_x, dose_y)
                    

        
        return losses
        

class MSE_and_DVH_loss(nn.Module):
    def __init__(self, DVH_kwargs, MSE_kwargs, weight_mse=1, weight_dvh=1, channel_ptv_oar=None,
                 DVH_class=value_DVH_Loss):
        """ #FIXME: change description
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(MSE_and_DVH_loss, self).__init__()

        self.weight_dvh = weight_dvh
        self.weight_mse = weight_mse

        self.channel_ptv_oar=channel_ptv_oar
        self.MSE = MSELoss(**MSE_kwargs)
        self.DVH = DVH_class(**DVH_kwargs)
        


    def forward(self, net_output: torch.Tensor, target: torch.Tensor,data: torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print("net_output.shape",net_output.shape,"target.shape",target.shape,"data.shape",data.shape)
    
        if self.DVH is value_DVH_Loss:
            print("set channel_ptv_OAR to None")
            self.channel_ptv_oar=None
        dvh_loss = self.DVH(net_output, target, mask_index=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar),channel_ptv_oar=self.channel_ptv_oar) \
            if self.weight_dvh != 0 else 0
        del data
        
        mse_loss = self.MSE(net_output, target) \
            if self.weight_mse != 0 else 0

       
        return self.weight_mse * mse_loss+ self.weight_dvh * dvh_loss
class MSE_and_2DVH_loss(nn.Module):
    def __init__(self, DVH_kwargs, MSE_kwargs, weight_mse=1, weight_dvh=1, channel_ptv_oar=None):
        """ #FIXME: change description
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(MSE_and_2DVH_loss, self).__init__()

        self.weight_dvh = weight_dvh
        self.weight_mse = weight_mse

        self.channel_ptv_oar=channel_ptv_oar
        self.MSE = MSELoss(**MSE_kwargs)
        self.vDVH = value_DVH_Loss(**DVH_kwargs)
        self.cDVH =criteria_DVH_Loss(**DVH_kwargs)
        


    def forward(self, net_output: torch.Tensor, target: torch.Tensor,data: torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print("net_output.shape",net_output.shape,"target.shape",target.shape,"data.shape",data.shape)
        c_ptv_oar,v_ptv_oar=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar,return_vptv_oar=True)
        
        cdvh_loss = self.cDVH(net_output, target, mask_index=c_ptv_oar,channel_ptv_oar=self.channel_ptv_oar) \
            if self.weight_dvh != 0 else 0  
               
           
        vdvh_loss = self.vDVH(net_output, target, mask_index=v_ptv_oar,channel_ptv_oar=None) \
            if self.weight_dvh != 0 else 0        
        
        mse_loss = self.MSE(net_output, target) \
            if self.weight_mse != 0 else 0

       
        return self.weight_mse * mse_loss+ self.weight_dvh * vdvh_loss+ self.weight_dvh * cdvh_loss
class MAE_and_DVH_loss(nn.Module):
    def __init__(self, DVH_kwargs, MAE_kwargs, weight_mae=1, weight_dvh=1, channel_ptv_oar=None,
                 DVH_class=value_DVH_Loss):
        """ #FIXME: change description
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(MAE_and_DVH_loss, self).__init__()

        self.weight_dvh = weight_dvh
        self.weight_mae = weight_mae

        self.channel_ptv_oar=channel_ptv_oar
        self.MAE = L1Loss(**MAE_kwargs)
        self.DVH = DVH_class(**DVH_kwargs)
        


    def forward(self, net_output: torch.Tensor, target: torch.Tensor,data: torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print("net_output.shape",net_output.shape,"target.shape",target.shape,"data.shape",data.shape)
    
        if self.DVH is value_DVH_Loss:
            print("set channel_ptv_OAR to None")
            self.channel_ptv_oar=None
        dvh_loss = self.DVH(net_output, target, mask_index=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar),channel_ptv_oar=self.channel_ptv_oar) \
            if self.weight_dvh != 0 else 0
        del data
        
        mae_loss = self.MAE(net_output, target) \
            if self.weight_mae != 0 else 0

       
        return self.weight_mae * mae_loss+ self.weight_dvh * dvh_loss
class MAE_and_2DVH_loss(nn.Module):
    def __init__(self, DVH_kwargs, MAE_kwargs, weight_mae=1, weight_dvh=1, channel_ptv_oar=None):
        """ #FIXME: change description
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(MAE_and_2DVH_loss, self).__init__()

        self.weight_dvh = weight_dvh
        self.weight_mae = weight_mae

        self.channel_ptv_oar=channel_ptv_oar
        self.MAE = L1Loss(**MAE_kwargs)
        self.vDVH = value_DVH_Loss(**DVH_kwargs)
        self.cDVH =criteria_DVH_Loss(**DVH_kwargs)
        


    def forward(self, net_output: torch.Tensor, target: torch.Tensor,data: torch.Tensor=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #print("net_output.shape",net_output.shape,"target.shape",target.shape,"data.shape",data.shape)
        c_ptv_oar,v_ptv_oar=get_roi_indexs(data, channel_ptv_oar=self.channel_ptv_oar,return_vptv_oar=True)
        
        cdvh_loss = self.cDVH(net_output, target, mask_index=c_ptv_oar,channel_ptv_oar=self.channel_ptv_oar) \
            if self.weight_dvh != 0 else 0  
               
           
        vdvh_loss = self.vDVH(net_output, target, mask_index=v_ptv_oar,channel_ptv_oar=None) \
            if self.weight_dvh != 0 else 0        
        
        mae_loss = self.MAE(net_output, target) \
            if self.weight_mae != 0 else 0

       
        return self.weight_mae * mae_loss+ self.weight_dvh * vdvh_loss+ self.weight_dvh * cdvh_loss
def get_roi_indexs(x:torch.Tensor,return_label=False, channel_ptv_oar=[0,1],return_vptv_oar=False):
    with torch.no_grad():
        device=x.device
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

    #     device=x.device
    #     x=torch.split(x,1,dim=1)
    #     if return_label:
    #         labels=[]
    #     if channel_ptv_oar==None:
    #         indexs=[]
    #         for c in range(0,len(x)):
    #             temp=x[c].flatten()
    #             label=temp.unique()
    #             if return_label:
    #                 labels.append(label)
    #             for l in label:
    #                 if l ==0: 
    #                     continue
    #                 where=(temp == l).nonzero().flatten().to(device)
    #                 indexs.append(where)
            
    #     else:
    #         # FIXME: change to this after training
    #         indexs=[]
    #         v_ptv_oar=[]
    #         for c in range(0,len(x)):
    #             index=[]#
    #             temp=x[c].flatten()
    #             label=temp.unique()
    #             if return_label:
    #                 labels.append(label)
    #             for l in label:
    #                 if l ==0: 
    #                     continue
    #                 where=(temp == l).nonzero().flatten().to(device)
    #                 index.append(where)#
    #                 v_ptv_oar.append(where)
    #             indexs.append(index)
 
        
    # if return_vptv_oar:
    #     return indexs,v_ptv_oar       
    # if return_label:
    #     return indexs,labels
    # return indexs   
# def get_roi_indexs(data:torch.Tensor, axes=None, square=False):

#     with torch.no_grad():

#         indexs = []
#         map=data#[:,c,...]

#         labels=torch.unique(map)
#         for label in labels:
#             if label!=0:
#                 where=(map == label).nonzero(as_tuple=False)
#                 indexs.append(where)

#     return indexs



