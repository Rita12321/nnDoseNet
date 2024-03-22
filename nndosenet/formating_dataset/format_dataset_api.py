from nndosenet.dicom_bridge.dicom2nifti import DicomConverter,NiftiLabeler
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import multiprocessing
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
def create_folderlist(path,folder_list=[],overwrite=False):
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
        #         shutil.move(dcm,join(patient_path,"Dicom",type))   #TODO: not working 
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


def multi_processs(num_processes):
     #TODO: add multiprocessing
        
        ### default_preprocessor line 271
        # multiprocessing magic.
        
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for outfile, infiles, segfiles in zip(output_filenames_truncated, image_fnames, seg_fnames):
                r.append(p.starmap_async(self.run_case_save,
                                         ((outfile, infiles, segfiles, plans_manager, configuration_manager,
                                           dataset_json),)))
            remaining = list(range(len(output_filenames_truncated)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(output_filenames_truncated), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)       