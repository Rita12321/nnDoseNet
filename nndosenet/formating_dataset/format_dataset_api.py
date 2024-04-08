from nndosenet.dicom_bridge.dicom2nifti import DicomConverter,NiftiLabeler
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import multiprocessing
import glob

import shutil
import math as m
import pandas as pd
from rt_utils import RTStructBuilder
from nndosenet.paths import nnDose_preprocessed, nnDose_raw,nnDose_dicom
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
def snaity_check():
    #TODO: check if structure set is all good
    pass
           
def organize_dicom(DATA_FOLDER):


    """
    DATA_FOLDER=nnUNet_dicom/Dataset/Tr
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
    print(sorted(os.listdir(DATA_FOLDER)))
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
        print(' finish organizing the the data in dir: '+DATA_FOLDER)    
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
    print(sorted(os.listdir(DATA_FOLDER)))
    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        if not os.path.isdir(join(DATA_FOLDER,patient_folder)):
            continue
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"])
        niftis=glob.glob(join(patient_path,"*.nii.gz"))
        for nifti in niftis:
            shutil.move(nifti,join(patient_path,"Nifti",))        
    
        
        
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
        print(' finish organizing the the data in dir: '+DATA_FOLDER)  
def format_dataset_from_dicom(folder,num_processes=3):
    dataset_name=folder.split("/")[-1]

    patients=sorted(os.listdir(folder))
    
    for patient in patients:
        if not os.path.isdir(join(folder,patient)):
            continue
        DicomConverter(folder,patient_folder=patient).run()    
    
    
    
    # r = []
    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #     for dataset_name, patient_folder, tr_ts in zip(dataset_name, patinet, tr_ts):
    #         r.append(p.starmap_async(DicomConverter(dataset_name=dataset_name,patient_folder=patient_folder,tr_ts=tr_ts).run(),
    #                                     ()))
    #     remaining = list(range(len(output_filenames_truncated)))
    #     # p is pretty nifti. If we kill workers they just respawn but don't do any work.
    #     # So we need to store the original pool of workers.
    #     workers = [j for j in p._pool]
    #     with tqdm(desc=None, total=len(output_filenames_truncated), disable=self.verbose) as pbar:
    #         while len(remaining) > 0:
    #             all_alive = all([j.is_alive() for j in workers])
    #             if not all_alive:
    #                 raise RuntimeError('Some background worker is 6 feet under. Yuck.')
    #             done = [i for i in remaining if r[i].ready()]
    #             for _ in done:
    #                 pbar.update()
    #             remaining = [i for i in remaining if i not in done]
    #             sleep(0.1)     
                
                    
def format_dataset_from_nifti(folder,dataset_id,dataset_name=None,tr_ts="Tr"):
    id_list=[]
    if dataset_name==None:
        dataset_name=folder.split("/")[-1]
        
    dataset_name="Dataset"+dataset_id+"_"+dataset_name
    patients=sorted(os.listdir(folder))
    
    if not os.path.isdir(join(nnDose_raw,dataset_name)):
        os.makedirs(join(nnDose_raw,dataset_name))
        print("Create folder in nnUNet_raw: ",join(nnDose_raw,dataset_name))
        
    if not os.path.isdir(join(nnDose_raw,dataset_name,"images"+tr_ts)):
        os.makedirs(join(nnDose_raw,dataset_name,"images"+tr_ts))
    if not os.path.isdir(join(nnDose_raw,dataset_name,"labels"+tr_ts)):
        os.makedirs(join(nnDose_raw,dataset_name,"labels"+tr_ts))
  
    print("Labeling the contours...")
    for id,patinet in enumerate(patients):
        id_list.append(id+1)
        NiftiLabeler(folder,patient_folder=patinet,patient_id=id+1,dataset_name=dataset_name,dataset_id=dataset_id,tr_ts=tr_ts).run() 
    df = pd.DataFrame({'id': id_list,
                   'patient': patients})  
    df.to_csv(join(nnDose_raw,dataset_name,'patient'+tr_ts+'.csv'), index=False)
    if tr_ts=="Tr":
        write_dataset_json(patient_num=len(patients),dataset_name=dataset_name)
        shutil.copy2(join(os.path.dirname(folder),'structure.json'),join(nnDose_raw,dataset_name,"structure.json"))
    print("Finish ! All needed file is in ",join(nnDose_raw,dataset_name),"now.")
#TODO: 
#...copy ct and dose in NIftilabeler for now...or forever... idk
# def copy_ct(folder):
#     pass
# def copy_dose():
#     pass
def write_dataset_json(patient_num,dataset_name):
    dict={"labels": {"background": "0", "dose": ""}, 
          "channel_names": {"0": "CT", "1": "PTV", "2": "OAR", "3": "BODY"}, 
          "numTraining": patient_num, 
          "file_ending": ".nii.gz",
          "mask_channel":[1,2,3]}
    json_object = json.dumps(dict, indent=4)

    with open(join(nnDose_raw,dataset_name,"dataset.json"), "w") as outfile:
        outfile.write(json_object)

def multi_processs(num_processes):
     #TODO: add multiprocessing
        
        ### default_preprocessor line 271
        # multiprocessing magic.
        
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        
        # compute_dose_metrics or compute_metrics(for segmentation)
        if dose_label:
            results = pool.starmap(
                compute_dose_metrics,
                list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                        [ignore_label] * len(files_pred)))
            )
        else:
            results = pool.starmap(
                compute_metrics,
                list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                        [ignore_label] * len(files_pred)))
            )