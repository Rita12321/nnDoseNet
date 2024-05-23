import argparse
import multiprocessing
import pandas as pd
import numpy as np
import glob
import os
from batchgenerators.utilities.file_and_folder_operations import join,isdir,isfile,save_json
from nndosenet.paths import nnDose_prediction,nnDose_dicom
from nndosenet.dicom_bridge.nifti2dicom import dose_to_dicom
from nndosenet.evaluation.gamma_analysis import gamma_analysis
from nndosenet.utilities.dataset_name_id_conversion import convert_id_to_dataset_name, maybe_convert_to_dataset_name
from tqdm import tqdm
from time import sleep
def nii2dcm_and_gamma(ref_path,pred_path,mask_path,save_folder,gamma=True) -> np.ndarray:
    
    #TODO: find the save path based on nnDoseNet_predict path? 
    name=pred_path.split("/")[-1]
    dicom_save_path=join(save_folder,"PredDicom",name+".dcm")
    
    
    ref_dcm,pred_dcm=dose_to_dicom(predict_nifti_path=pred_path,reference_dicom_path=ref_path,save_path=dicom_save_path,for_gamma=True)
    if gamma:
        gamma_save_path=join(save_folder,"Gamma",name+".nii.gz")
        gamma_options,pass_gamma_num,mask_num=gamma_analysis(reference=ref_dcm,evaluation=pred_dcm,mask_path=mask_path,save_path=gamma_save_path)
        dict={"gamma_setting":gamma_options,
              "passing_rate":pass_gamma_num*100/mask_num,
              "pass_gamma_num":pass_gamma_num,
              "mask_num":mask_num,
              "prediction_file":pred_path,
              "reference_file":ref_path,
              "Body/dose mask_file":mask_path
        }
        
        return dict

def apply_nii2dcm_and_gamma_to_folder(prediction_folder: str,
                                   reference_folder:str,
                                   output_folder: str,
                                   patient_csv:str,
                                   num_processes=8,gamma=True) -> None:
     

    if gamma:
        if not os.path.isdir(join(output_folder,"Gamma")):
            os.mkdir(join(output_folder,"Gamma"))
    if not os.path.isdir(join(output_folder,"PredDicom")):
        os.mkdir(join(output_folder,"PredDicom"))        
    # maybe_mkdir_p(output_folder)
    
    patient_csv=pd.read_csv(patient_csv)
    ref_list=list(patient_csv['patient'])
    pred_list=list(patient_csv['id'])

    #TODO: change Body.nii.gz by using structure.json
    r=[]
    results_dict ={}# open exist one
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for i,pred in enumerate(pred_list):
            r.append(p.starmap_async(nii2dcm_and_gamma,
                                     ((join(reference_folder, ref_list[i],"Dicom","RD",os.listdir(join(reference_folder, ref_list[i],"Dicom","RD"))[0]),
                                       join(prediction_folder, str(pred).zfill(4)),
                                       join(reference_folder, ref_list[i],"Nifti","Body.nii.gz"),
                                       output_folder,gamma),)))
            
        remaining = list(range(len(pred_list)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
        workers = [j for j in p._pool]
        with tqdm(desc="", total=len(pred_list) )as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                done = [i for i in remaining if r[i].ready()]
                for i in done:
                    pbar.update()
                    results_dict.update(r[i].get()[0])
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
    #TODO: combine summaries
    pass_list=[]
    for case in results_dict:
        pass_list.append(case["passing_rate"])
    mean_dict={"gamma_option":results_dict[0]["gamma_setting"],
               "mean_gamma_passing_rate":np.mean(pass_list),
               "std_gamma_passing_rate":np.std(pass_list),
               "total_case":len(results_dict)
    }
    results_dict = mean_dict+dict(sorted(results_dict.items()))
    save_json(results_dict,join(output_folder,"gamma_summary.json"),sort_keys=False)

def entry_point_nii2dcm_and_gamma_analysis():
    parser = argparse.ArgumentParser('Convert nifti prediction to dicom if the ground truth is in dicom format. \
                                        Also perform gamma analysis on prediction in default ')
    
    parser.add_argument('-id', type=str, required=False, help='Prediction id')
    parser.add_argument('-name', type=str, required=False, help='Dataset Name')
    
    parser.add_argument('-pred', type=str, required=False, default=None,help='Prediction folder, default use\
                        "[nndose_prediction]/[-name]/Dataset+[-id]}/Prediction/[all_configure_and esemble folder]" that is created format and creat when formatting datset Ts')
    parser.add_argument('-ref', type=str, required=False, default=None, help='Reference folder')
    parser.add_argument('-out', type=str, required=False, default=None, help='Output folder')
    parser.add_argument('-csv', type=str, required=False, default=None, help='csv convertion list of petient_num nifti folder and patient folder name in dicom folder, default use the \
                        "[nndose_prediction]/[-name]/Dataset+[-id]}/GroundTruth/patientTs.csv" that is created format and creat when formatting datset Ts')
    parser.add_argument('-np', type=int, required=False, default=3,
                        help=f"Number of processes used for dicom convertion and gamma analysis. !! Pease set this nps less than \
                            nps in other process. Gamma analysis required LOTS of Ram so it may crash if you have too many nps , Default: 3")
    parser.add_argument('--no_gamma', action='store_false',
                        help='Only convert prediction of nifti back to dicom without doing gamma analysis. It would be wayyyy faster if you do not need gamma!')
    args = parser.parse_args()
    
    ## prediction folder default 
    prediction_folder=args.pred
    if prediction_folder is None:
        if args.name is None or args.id is None:
            raise(RuntimeError("Either input predict folder or input dataset name and id"))
        prediction_folder=join(nnDose_prediction,args.name,"Dataset"+args.id+"_"+args.name,"Prediction")
        print("default prediction folder: ", prediction_folder)
        
    ## reference folder default    
    reference_folder=args.ref
    if reference_folder is None:
        reference_folder=join(nnDose_dicom,args.name,"Ts")
        print("default reference folder: ", reference_folder)
    
    if not os.path.isdir(reference_folder):
        raise(RuntimeError("Reference_folder: "+reference_folder+" do not exist"))  
    
    ## patient csv default  
    patient_csv=args.csv
    if patient_csv is None:
        patient_csv=join(nnDose_prediction,args.name,"Dataset"+args.id+"_"+args.name,"GroundTruth","patientTs.csv")
        print("default patient_csv: ", patient_csv)
    if not os.path.isfile(patient_csv):
        raise(RuntimeError("Patient_csv: "+patient_csv+" do not exist"))      
    #TODO: else glob csv
    
    ## output folder default
    output_folder=args.out

    if args.pred==None:
        print("didn't in put prediction folder, run default, all configuration and out put folder would be auto assigned")
        for config in os.listdir(prediction_folder):
            if os.path.isdir(join(prediction_folder,config)):
                print("runnnig: ", config)
                output_folder=join(prediction_folder,config)
                print("output_folder: ", output_folder)
                apply_nii2dcm_and_gamma_to_folder(join(prediction_folder,config), reference_folder, output_folder, patient_csv, args.np)
                print("all done")
            
    else:
        if isfile(os.listdir(prediction_folder)[0]): #input folder with file 
            apply_nii2dcm_and_gamma_to_folder(prediction_folder, reference_folder, output_folder, patient_csv, args.np)
        elif isdir(os.listdir(prediction_folder)[0]): #input folder with folders
            try:
                for config in os.listdir(prediction_folder):
                    if os.path.isdir(join(prediction_folder,config)):
                        print("runnnig: ", config)
                        if output_folder is None:
                            output_folder=join(prediction_folder,config)
                            print("default output folder: ", output_folder)
                        apply_nii2dcm_and_gamma_to_folder(join(prediction_folder,config), reference_folder, output_folder, patient_csv, args.np)
                        print("all done")
            except:
                print("-pred should be a folder contain niftis, I tried to run all the folder in it, but fail:((")
    
    print("finish !!! ")
    
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
    
    # if output_folder is None:
    #     output_folder=os.path.dirname(prediction_folder)
    
    if not os.path.isdir(output_folder):
        raise(RuntimeError("Output_folder: "+output_folder+" do not exist")) 
    apply_nii2dcm_and_gamma_to_folder(prediction_folder, reference_folder, output_folder, patient_csv, 8)