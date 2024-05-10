import SimpleITK as sitk   
import numpy as np  
    
    
def difference_map(pred_np,label_np,dose_mask=None,MSE=True,MAE=True): 
    diff_map=label_np-pred_np
    num_dose=None
    MSE=None
    MAE=None
 
    if dose_mask != None:
        diff_map=diff_map*dose_mask
        num_dose=np.count_nonzero(dose_mask.flatten())
    if MSE:
        MSE=MSE_from_diff(diff_map=diff_map,num_dose=num_dose)
    if MAE:
        MAE=MAE_from_diff(diff_map=diff_map,num_dose=num_dose)
    return diff_map,MAE,MSE

def MSE_from_diff(diff_map,num_dose=None):
    if num_dose !=None:
        if num_dose.isnumeric():
            return np.sum(np.square(diff_map.flatten()))/num_dose
   
    return np.mean(np.sum(np.square(diff_map.flatten())))
    
def MAE_from_diff(diff_map,num_dose=None):
    if num_dose !=None:
        if num_dose.isnumeric():
            return np.sum(np.abs(diff_map.flatten()))/num_dose
    
    return np.mean(np.sum(np.abs(diff_map.flatten())))    
#TODO: difference map api