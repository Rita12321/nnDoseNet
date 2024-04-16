import argparse
import multiprocessing
import shutil
from multiprocessing import Pool
# from typing import Union, Tuple, List, Callable
import pandas as pd
import numpy as np
import glob
import os
# from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, maybe_mkdir_p, join, isfile, \
    isdir, save_pickle, load_pickle, save_json
from nndosenet.configuration import default_num_processes
# from nndosenet.evaluation.accumulate_cv_results import accumulate_cv_results
# from nndosenet.evaluation.evaluate_predictions import region_or_label_to_mask, compute_dose_metrics_on_folder, \
#     load_summary_json, label_or_region_to_key
# from nndosenet.imageio.base_reader_writer import BaseReaderWriter
from nndosenet.paths import nnDose_raw,nnDose_prediction,nnDose_results,nnDose_dicom
# from nndosenet.utilities.file_path_utilities import folds_tuple_to_string
# from nndosenet.utilities.json_export import recursive_fix_for_json_export
# from nndosenet.utilities.plans_handling.plans_handler import PlansManager
from nndosenet.dicom_bridge.nifti2dicom import dose_to_dicom
from nndosenet.postprocessing.gamma_analysis import gamma_analysis

def nii2dcm_and_gamma(ref_path,pred_path,save_folder) -> np.ndarray:
    
    #TODO: find the save path based on nnDoseNet_predict path? 
    name=pred_path.split("/")[-1]
    dicom_save_path=join(save_folder,"PredDicom",name+".dcm")
    gamma_save_path=join(save_folder,"Gamma",name+".nii.gz")
    
    ref_dcm,pred_dcm=dose_to_dicom(predict_nifti_path=pred_path,reference_dicom_path=ref_path,save_path=dicom_save_path,for_gamma=True)
    gamma_analysis(reference=ref_dcm,evaluation=pred_dcm,save_path=gamma_save_path)


def apply_postprocessing_to_folder(prediction_folder: str,
                                   reference_folder:str,
                                   output_folder: str,
                                   patient_csv:str,
                                   num_processes=8) -> None:
     

  
    if not os.path.isdir(join(output_folder,"Gamma")):
        os.mkdir(join(output_folder,"Gamma"))
    if not os.path.isdir(join(output_folder,"PredDicom")):
        os.mkdir(join(output_folder,"PredDicom"))        
    # maybe_mkdir_p(output_folder)
    
    patient_csv=pd.read_csv(patient_csv)
    ref_list=list(patient_csv['patient'])
    pred_list=list(patient_csv['id'])
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:

        _ = p.starmap(nii2dcm_and_gamma,zip(
                          [join(reference_folder, i,"Dicom","RD",os.listdir(join(reference_folder, i,"Dicom","RD"))[0]) for i in ref_list],
                          [join(prediction_folder, str(i).zfill(4)) for i in pred_list],
                          [output_folder] * len(pred_list))
                      )
        #TODO: apply evaluation or read and write _evaluation 



def entry_point_apply_postprocessing():
    parser = argparse.ArgumentParser('Apples postprocessing specified in pp_pkl_file to input folder.')
    
    parser.add_argument('-id', type=str, required=False, help='Prediction id')
    parser.add_argument('-name', type=str, required=False, help='Dataset Name')
    
    parser.add_argument('-pred', type=str, required=False, default=None,help='Prediction folder, default use\
                        "[nndose_prediction]/[-name]/Dataset+[-id]}/Prediction" that is created format and creat when formatting datset Ts')
    parser.add_argument('-ref', type=str, required=False, default=None, help='Reference folder')
    parser.add_argument('-out', type=str, required=False, default=None, help='Output folder')
    
    parser.add_argument('-csv', type=str, required=False, default=None, help='csv convertion list of petient_num nifti folder and patient folder name in dicom folder, default use the \
                        "[nndose_prediction]/[-name]/Dataset+[-id]}/GroundTruth/patientTs.csv" that is created format and creat when formatting datset Ts')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f"number of processes to use. Default: {default_num_processes}")

    args = parser.parse_args()
    
    ## prediction folder default 
    prediction_folder=args.pred
    if prediction_folder is None:
        if args.name is None or args.id is None:
            raise(RuntimeError("Either input predict folder or input dataset name and id"))
        prediction_folder=join(nnDose_prediction,args.name,"Dataset"+args.id,"Prediction")
        
    if not os.path.isdir(prediction_folder):
        raise(RuntimeError("Prediction folder: "+prediction_folder+" do not exist"))
    elif len(os.listdir(prediction_folder))<1:
        raise(RuntimeError("There is no files in Prediction folder: "+prediction_folder))
    elif len(glob.glob(join(prediction_folder,"*.nii.gz")))<1:
        raise(RuntimeError("There is no nifti file in Prediction folder: "+prediction_folder))
    
    ## reference folder default    
    reference_folder=args.ref
    if reference_folder is None:
        reference_folder=join(nnDose_dicom,args.name,"Tr")
    
    if not os.path.isdir(reference_folder):
        raise(RuntimeError("Reference_folder: "+reference_folder+" do not exist"))  
    
    ## patient csv default  
    patient_csv=args.patientcsv
    if patient_csv is None:
        patient_csv=join(nnDose_prediction,args.name,"GroundTruth","patientTs.csv")
    if not os.path.isfile(patient_csv):
        raise(RuntimeError("Patient_csv: "+patient_csv+" do not exist"))      
    #TODO: else glob csv
    
    ## output folder default
    output_folder=args.out
    if output_folder is None:
        output_folder=os.path.dirname(prediction_folder)
    
    if not os.path.isdir(output_folder):
        raise(RuntimeError("Output_folder: "+output_folder+" do not exist"))   
    
    apply_postprocessing_to_folder(prediction_folder, reference_folder, output_folder, patient_csv, args.np)

if __name__ == '__main__':
    ## reference folder default    
    reference_folder=None
    prediction_folder=None
    output_folder=None
    patient_csv=None
    

    name="prostate_UAB"
    id="520"
        
    ## prediction folder default 

    if prediction_folder is None:
        if name is None or id is None:
            raise(RuntimeError("Either input predict folder or input dataset name and id"))
        prediction_folder=join(nnDose_prediction,name,"Dataset"+id,"Prediction")
        
    if not os.path.isdir(prediction_folder):
        raise(RuntimeError("Prediction folder: "+prediction_folder+" do not exist"))
    elif len(os.listdir(prediction_folder))<1:
        raise(RuntimeError("There is no files in Prediction folder: "+prediction_folder))
    elif len(glob.glob(join(prediction_folder,"*.nii.gz")))<1:
        raise(RuntimeError("There is no nifti file in Prediction folder: "+prediction_folder))
    
    ## reference folder default    

    if reference_folder is None:
        reference_folder=join(nnDose_dicom,name,"Ts")
    if not os.path.isdir(reference_folder):
        raise(RuntimeError("Reference_folder: "+reference_folder+" do not exist"))      

    
    ## patient csv default  

    if patient_csv is None:
        patient_csv=join(nnDose_prediction,name,"Dataset"+id,"GroundTruth","patientTs.csv")
    if not os.path.isfile(patient_csv):
        raise(RuntimeError("Patient_csv: "+patient_csv+" do not exist"))      
    #TODO: else glob csv
    
    ## output folder default
    
    if output_folder is None:
        output_folder=os.path.dirname(prediction_folder)
    
    if not os.path.isdir(output_folder):
        raise(RuntimeError("Output_folder: "+output_folder+" do not exist")) 
    apply_postprocessing_to_folder(prediction_folder, reference_folder, output_folder, patient_csv, 8)