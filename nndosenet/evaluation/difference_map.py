import SimpleITK as sitk   
import numpy as np  
import multiprocessing  
import os 
import pandas as pd
# from nndosenet.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
#     determine_reader_writer_from_file_ending
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
import tqdm

def difference_map(label_np,pred_np,dose_mask=None,do_MSE=False,do_MAE=False): 
    diff_map=label_np-pred_np
    num_dose=None
    MSE=None
    MAE=None
 
    if dose_mask != None:
        diff_map=diff_map*dose_mask
        num_dose=np.count_nonzero(dose_mask.flatten())
    if do_MSE:
        MSE=MSE_from_diff(diff_map=diff_map,num_dose=num_dose)
    if do_MAE:
        MAE=MAE_from_diff(diff_map=diff_map,num_dose=num_dose)
    #print(np.max(diff_map),np.max(pred_np),np.max(label_np))
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
def write_diffmap(label,pred,file_ending,out_folder,dose_mask=None,do_MSE=False,do_MAE=False):
       #todo imageio file ending
    #print(label,pred)
    label_img=sitk.ReadImage (label)
    label_np=sitk.GetArrayFromImage(label_img)
    pred_np=sitk.GetArrayFromImage(sitk.ReadImage (pred))
    diffmap,_,_=difference_map(label_np,pred_np,dose_mask,do_MSE,do_MAE)
    if dose_mask is None:
        dose_mask=label_np>0
  
    diffmap = np.where(dose_mask >=1, diffmap, 160) #0. set 160 becuse slice is anoying, 

    diff_img=sitk.GetImageFromArray(diffmap)
    diff_img.CopyInformation(label_img)
    sitk.WriteImage(diff_img,join(out_folder,label.split("/")[-1]))

def diff_map_api(label_folder,pred_folder,file_ending,out_folder,num_processes): 
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
        
    labels=sorted(subfiles(label_folder,suffix=file_ending))
    preds=sorted(subfiles(pred_folder,suffix=file_ending))
     #todo add progress bar
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        _ = p.starmap(write_diffmap,zip(
                          labels,
                          preds,
                          [file_ending] * len(labels),
                          [out_folder] * len(labels))
                      )
 
    
def entrypoint_diff_map():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt_folder', type=str, help='folder with ground truth label')
    parser.add_argument('-pred_folder', type=str, help='folder with predicted label')
    parser.add_argument('-djfile', type=str, required=False,default=None,
                        help='dataset.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: [pred_folder]/DiffMap')
    parser.add_argument('-np', type=int, required=False, default=8,
                        help=f'number of processes used. Optional. Default: 8')
    args = parser.parse_args()
    
    if args.djfile is None:
        file_ending=".nii.gz"
    else:
        file_ending=load_json(args.djfile)["file_ending"]
        if file_ending != ".nii.gz":
            print("nnDoseNet only support nifti file for now :(")
            file_ending=".nii.gz"
    if args.o is None:
        out_folder=join(args.pred_folder,"DiffMap")
    else:
        out_folder=args.o
        
    diff_map_api(args.gt_folder,args.pred_folder,file_ending=file_ending,out_folder=out_folder,num_processes=args.np)
# if __name__ == '__main__':
    # gt_folder="/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/prostate_UAB/Dataset099_prostate_UAB/GroundTruth"
    # pred_folder="/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/prostate_UAB/Dataset099_prostate_UAB/Prediction/2d"
    
    # diff_map_api(gt_folder,pred_folder,djfile,out_folder,num_processes=8)