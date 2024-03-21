from dicom_bridge.dicom2nifti import DicomConverter,NiftiLabeler
import pydicom
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import threading
import glob

import shutil
import math as m
from rt_utils import RTStructBuilder
from nndosenet.paths import nnUNet_preprocessed, nnUNet_raw,nnUNet_dicom
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
def create_folderlist(path,folder_list=[],overwrite="False"):
    for sub_folder in folder_list:
        if not os.path.isdir(join(path,sub_folder)):
            os.makedirs(join(path,sub_folder))
        elif overwrite:
            shutil.rmtree(join(path,sub_folder))
            os.makedirs(join(path,sub_folder))
def snaity_check():
    #TODO: check if structure set
    pass
           
def organize_dicom(DATA_FOLDER):


    """
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

    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"],overwrite=True)
        create_folderlist(patient_path,folder_list=["Nifti","Dicom"])
        type_list=["CT","RD","RS"]
        create_folderlist(join(patient_path,"Dicom"),folder_list=type_list)
        for type in type_list:
            dcms=glob.glob(join(patient_path,type+"*.dcm"))
            if len(dcms)<=0:
                print('WARNING: cannot find any file start with" '+type+'".dcm'\
                        'trying to find files not ending with .dcm')
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
        type="extra"
        create_folderlist(join(patient_path,"Dicom",type))  
        dcms=glob.glob(join(patient_path,"*"))      
        if len(dcms)>4:
            print("There are ", len(dcms)-4, "extra files, that is not starting with CT,RD or RS. They are moved to folder extra")
            for dcm in dcms:
                if os.path.isdir(dcm):
                    continue
                shutil.move(dcm,join(patient_path,"Dicom",type))    
        print(' finish organizing the the data in dir: '+DATA_FOLDER)    

def format_dataset_from_dicom(folder,tr_ts="Tr"):
    dataset_name=folder.split("/")[-1]
    organize_dicom(folder)
    patients=os.listdir(folder)
    for patinet in patients:
        DicomConverter(dataset_name=dataset_name,patient_folder=patinet,tr_ts=tr_ts)    
        
def format_dataset_from_nifti(folder,dataset_id,tr_ts="Tr"):
    id_list=[]
    dataset_name=folder.split("/")[-1]
    patients=sorted(os.listdir(folder))
    for id,patinet in enumerate(patients):
        id_list.append(id)
        NiftiLabeler(patient_folder=patinet,patient_id=id,dataset_name=dataset_name,dataset_id=dataset_id,tr_ts=tr_ts) 
    #TODO:
    #save_csv({id_list,patients},join())

def write_dataset_json():
    pass
       