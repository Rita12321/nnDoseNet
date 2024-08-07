from nndosenet.dicom_bridge.dicom2nifti import DicomConverter,NiftiLabeler
import numpy as np
import SimpleITK as sitk
# import nibabel as nib
import os
import multiprocessing
import glob
from tqdm import tqdm
from time import sleep
import shutil
# import math as m
import pandas as pd
# from rt_utils import RTStructBuilder
from nndosenet.paths import nnDose_preprocessed, nnDose_raw,nnDose_dicom,nnDose_prediction
from batchgenerators.utilities.file_and_folder_operations import *


def sitk_read_dcm(in_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    dicom = reader.Execute()
    return dicom 

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
#TODO: change to find basefolder method
def create_folderlist(path,folder_list=[],overwrite=False):
    for sub_folder in folder_list:
        if not os.path.isdir(join(path,sub_folder)):
            os.makedirs(join(path,sub_folder))
        elif overwrite:
            shutil.rmtree(join(path,sub_folder))
            os.makedirs(join(path,sub_folder))
            
def sanity_check_match_roi():
    #TODO: load match roi summary if none check if structure set is all good
    pass
def sanity_check_dcm_find():
    pass          
def organize_dicom(DATA_FOLDER):

    """
    DATA_FOLDER=nnDose_dicom/Dataset/Tr
    CT for CT image, RS for structure of image, RD or dose map (just 1 RD file please)
    The data structure of DATA_FOLDER should be like this
    -DATA_FOLDER
        -patient 1
            -CT.12345678.91011112.dcm
            -CT.12345678.91011113.dcm
            -CT.12345678.91011114.dcm
            ...
            -RS.12345678.91011117.dcm
            -RS.12345678.91011118.dcm
            ...
            -RD.12345678.91011119.dcm
        -patient 2
            -CT.22345678.91011112.dcm
            -CT.22345678.91011113.dcm
            -CT.22345678.91011114.dcm
            ...
            -RS.22345678.91011117.dcm
            -RS.22345678.91011118.dcm
            ...
            -RD.22345678.91011119.dcm
        -patient 3 
        ...
        
    After apply organize_dicom DATA_FOLDER should be like this
    -DATA_FOLDER
        -0001
            -Nifti
            -Dicom
                -CT
                    -CT.12345678.91011112.dcm
                    -CT.12345678.91011113.dcm
                    -CT.12345678.91011114.dcm
                    ...
                -RS
                    -RS.12345678.91011117.dcm
                    -RS.12345678.91011118.dcm
                    ...
                -RD
                    -RD.12345678.91011119.dcm
                -extra
                    ....
                
        -0002
            -Nifti
            -Dicom
                -CT
                    -CT.22345678.91011112.dcm
                    -CT.22345678.91011113.dcm
                    -CT.22345678.91011114.dcm
                    ...
                -RS
                    -RS.22345678.91011117.dcm
                    -RS.22345678.91011118.dcm
                    ...
                -RD
                    -RD.22345678.91011119.dcm
                -extra
                    ....
        -0003 
        ...
    """
    #print(sorted(os.listdir(DATA_FOLDER)))
    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        if not os.path.isdir(join(DATA_FOLDER,patient_folder)):
            continue
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti","Dicom"])
        type_list=["CT","RD","RS"]
        create_folderlist(join(patient_path,"Dicom"),folder_list=type_list)
        for type in type_list:
            dcms=glob.glob(join(patient_path,type+"*.dcm"))
            if len(dcms)<=0:
                # print('WARNING: cannot find any file start with" '+type+'".dcm'\
                #         'trying to find files not ending with .dcm')
                dcms=glob.glob(join(patient_path,type+"*"))
                if len(dcms)<=0:
                    print('WARNING: cannot find any file start with" '+type+'" '\
                            'please check the if the folder directory is correct or it has" '+type+' " in file name.') 
                else:
                    print('find',len(dcms),'file start with " ',type,
                        'please make sure it end with .dcm next time, otherwise there might have error cause by it') 
            for dcm in dcms:
                shutil.move(dcm,join(patient_path,"Dicom",type))        
        
        
        
        ##extra
        # type="extra"
        # create_folder(join(patient_path,"Dicom",type))  
        # dcms=glob.glob(join(patient_path,"*"))      
        # if len(dcms)>4:
        #     print("There are ", len(dcms)-4, "extra files, that is not starting with CT,RD or RS. They are moved to folder extra")
        #     for dcm in dcms:
        #         if os.path.isdir(dcm):
        #             continue
        #         shutil.move(dcm,join(patient_path,"Dicom",type))  
        #print(' finish organizing the the data in dir: '+DATA_FOLDER)    
def organize_nifti(DATA_FOLDER):
    """
    DATA_FOLDER=nnUNet_dicom/Dataset/Tr
    After apply organize_nifti DATA_FOLDER should be like this
    -DATA_FOLDER
        -0001
            -Nifti
                -ct.nii.gz
                -dose.nii.gz
                -structure1.nii.gz
                -structure2.nii.gz
                -structure3.nii.gz
            -Dicom
                
        -0002
            -Nifti
                -ct.nii.gz
                -dose.nii.gz
                -structure1.nii.gz
                -structure2.nii.gz
                -structure3.nii.gz
            -Dicom
        -0003 
        ...
    """
    #print(sorted(os.listdir(DATA_FOLDER)))
    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        if not os.path.isdir(join(DATA_FOLDER,patient_folder)):
            continue
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"])
        niftis=glob.glob(join(patient_path,"*.nii.gz"))
        for nifti in niftis:
            shutil.move(nifti,join(patient_path,"Nifti",))        
 
    print(' finish organizing the the data in dir: '+DATA_FOLDER)  
def dicom_convert(folder,patient,structure_json=None):
    temp=DicomConverter(folder,patient_folder=patient,structure_name=structure_json).run()
    return {patient:temp}
    
def format_dataset_from_dicom(folder,tr_ts,num_processes=3,structure_json=None):
    folder=join(folder,tr_ts)

    patients=sorted(os.listdir(folder))
    
    # for patient in patients:
    #     if not os.path.isdir(join(folder,patient)):
    #         continue
    #     DicomConverter(folder,patient_folder=patient).run()    
    r=[]
    results_dict ={}# open exist one


    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for patient in patients:
            r.append(p.starmap_async(dicom_convert,
                                     ((folder,patient,structure_json),)))
            
        remaining = list(range(len(patients)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
        workers = [j for j in p._pool]
        with tqdm(desc="converting Dicom2Nifti", total=len(patients) )as pbar:
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
    
    # for i in r:
    #     results_dict.update(i.get()[0]) #todo test it in while done , update there

    #print(results_dict)
    save_sum_name = os.path.join(os.path.dirname(folder),tr_ts+'_roi_summary.csv')#todo:  to base +trts
    results_dict = dict(sorted(results_dict.items()))
    dataFrame = pd.DataFrame(results_dict).T#.sort_index()
   
    # dataFrame=dataFrame.rename(columns={dataFrame.columns[0]:"patients"})
    # dataFrame.sort_values(by=['patients'])
    dataFrame.to_csv(save_sum_name)
    save_json(results_dict,save_sum_name.replace('.csv', '.json'))
    
def nifti_label(folder,patient_folder,patient_id,dataset_name,dataset_id,tr_ts,structure_label=None):
    NiftiLabeler(folder,patient_folder,patient_id,dataset_name,dataset_id,tr_ts,structure_label=structure_label).run()
                       
def format_dataset_from_nifti(folder,dataset_id,dataset_name_only=None,tr_ts="Tr",num_processes=8,structure_label=None):
    id_list=[]
    if dataset_name_only == None:
        dataset_name_only = folder.split("/")[-2]
        
    dataset_name="Dataset"+dataset_id+"_"+dataset_name_only
    
    patients = sorted(os.listdir(folder))
    id_list = [*range(1,len(patients)+1, 1)] 
    
    if os.path.isfile(os.path.join(os.path.dirname(folder),tr_ts+'_roi_summary.csv')):
        file = pd.read_csv(os.path.join(os.path.dirname(folder),tr_ts+'_roi_summary.csv'), usecols=[0,1],index_col=False)
        if "patient_id" in file.keys():
            patients = file[file.keys()[0]].tolist()
            id_list = file["patient_id"].tolist()

    
    if tr_ts=="Tr":
        save_folder=join(nnDose_raw,dataset_name)
    else: 
        save_folder=join(nnDose_prediction,dataset_name_only,dataset_name,"GroundTruth")
    if not os.path.isdir(join(save_folder)):
        os.makedirs(join(save_folder))
        print("Create folder in : ",join(save_folder))

    if not os.path.isdir(join(save_folder,"images"+tr_ts)):
        os.makedirs(join(save_folder,"images"+tr_ts))
    if not os.path.isdir(join(save_folder,"labels"+tr_ts)):
        os.makedirs(join(save_folder,"labels"+tr_ts))

    print("Labeling the contours")
    r=[]
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for patient,id in zip(patients,id_list):
            r.append(p.starmap_async(nifti_label,
                                     ((folder,patient,id,dataset_name,dataset_id,tr_ts,structure_label),)))
        remaining = list(range(len(patients)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
        workers = [j for j in p._pool]
        with tqdm(desc="labeling contours", total=len(patients) )as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
    
    df = pd.DataFrame({'id': id_list,
                   'patient': patients})  
    df.to_csv(join(save_folder,'patient'+tr_ts+'.csv'), index=False)
    
    if tr_ts=="Tr":
        write_dataset_json(patient_num=len(patients),dataset_name=dataset_name)
    if structure_label != None:
        shutil.copy2(join(structure_label),join(save_folder,"structure.json"))
    else:
        shutil.copy2(join(os.path.dirname(folder),'structure.json'),join(save_folder,"structure.json"))
  
            
    print("Finish ! All needed file is in ",join(save_folder),"now.")

def write_dataset_json(patient_num,dataset_name):
    dict={"labels": {"background": "0", "dose": ""}, 
          "channel_names": {"0": "CT", "1": "Target", "2": "OAR", "3": "BODY"}, 
          "numTraining": patient_num, 
          "file_ending": ".nii.gz",
          "mask_channel":[1,2,3]}
    json_object = json.dumps(dict, indent=4)

    with open(join(nnDose_raw,dataset_name,"dataset.json"), "w") as outfile:
        outfile.write(json_object)


                      
    
