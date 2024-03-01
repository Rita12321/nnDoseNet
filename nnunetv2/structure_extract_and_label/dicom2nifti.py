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
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw,nnUNet_dicom
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

def create_folderlist(path,folder_list=[],overwrite="False"):
    for sub_folder in folder_list:
        if not os.path.isdir(join(path,sub_folder)):
            os.makedirs(join(path,sub_folder))
        elif overwrite:
            shutil.rmtree(join(path,sub_folder))
            os.makedirs(join(path,sub_folder))
            
def organize_dicom(DATA_FOLDER=None):
    if DATA_FOLDER==None:
        DATA_FOLDER=join(nnUNet_dicom,"Tr")
    original_name=[]

    """
    CT for CT image, RS for structure of image, RD or dose map
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
        #print(patient_folder)
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"],overwrite=True)
        create_folderlist(patient_path,folder_list=["Nifti","Dicom"])
        type_list=["CT","RD","RS","extra"]
        create_folderlist(join(patient_path,"Dicom"),folder_list=type_list)
        for type in type_list:
            dcms=glob.glob(join(patient_path,type+"*"))
            for dcm in dcms:
                shutil.move(dcm,join(patient_path,"Dicom",type))
            if len(dcms<=0) : 
                    print('WARNING: cannot find the .dcm file contain" '+type+'" '\
                        'please check the if the folder directory is correct or it has" '+type+' " in file name.')             
        
        dcms=glob.glob(join(patient_path,"*"))
        
        if len(dcms)>4:
            print("There are ", len(dcms)-4, "extra dcm, which are the files, not starting with CT,RD or RS. They are moved to folder extra")
            type="extra"
            create_folderlist(join(patient_path,"Dicom",type))
            for dcm in dcms:
                if dcm in type_list:
                    continue
                shutil.move(dcm,join(patient_path,"Dicom",type))    
                    
        print(' finish organizing the the data in dir: '+DATA_FOLDER)           
class DicomConverter():#patient wise
    def __init__(self,dataset_name,patient_folder,patient_id,tr_ts="Tr"):      
        self.dataset_name=dataset_name 
        self.patient_folder=patient_folder
        self.tr_ts=tr_ts
        
        self.raw_dicom_folder = join(nnUNet_dicom,self.dataset_name,tr_ts,patient_folder)
        self.raw_dataset_folder = join(nnUNet_raw,self.dataset_name)

        self.structure_label = load_json(join(nnUNet_dicom,self.dataset_name, 'structure.json'))["structure_label"]
        self.structure_name = load_json(join(nnUNet_dicom,self.dataset_name, 'structure.json'))["structure_name"]
        
        self.ct_folder=join(self.raw_dicom_folder,"Dicom","CT")
        self.dose_folder=join(self.raw_dicom_folder,"Dicom","RD")
        self.struct_folder=join(self.raw_dicom_folder,"Dicom","RS")
        self.nii_folder=join(self.raw_dicom_folder,"Nifti")
        
        
  
    def get_ct(self,save=True):
        ct_path=join(self.nii_folder,"ct.nii.gz")
        if os.path.isfile(ct_path):
            return sitk.ReadImage(ct_path)
        
        if join(self.nii_folder,"img.nii.gz"):
            print('"ct.nii.gz" is not in Nifti folder but found "img.nii.gz".\n'\
                    'I am assuming img is the ct or at least the img you want to use as input\n'\
                    'so I am Renaming img.nii.gz to ct.nii.gz....\n'\
                    'if that is not what you want.... please start over and see the instruction')
            os.rename(join(self.nii_folder,"img.nii.gz"),join(self.nii_folder,"ct.nii.gz"))
            return sitk.ReadImage(ct_path)
        
        if len (os.listdir(self.ct_folder)>0):
            ct=sitk_read_dcm(self.ct_folder)
            if save:
                sitk.WriteImage(ct,ct_path)
            return ct
        
        raise RuntimeError("can't even find the dicom for ct, please organize everything as instruction")
    
    def get_dose(self,save=True):
        dose_path=join(self.nii_folder,"dose.nii.gz")
        if os.path.isfile(dose_path):
            return sitk.ReadImage(dose_path)
       
        if len(os.listdir(self.dose_folder))==1:
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                #print("converting dose",i," in",self.raw_dicom_folder)
                return self.convert_dose2nii(dose,id=i)
        if(len(os.listdir(self.dose_folder))>1):
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                print("WARNING! Multi Dose Plan Detected: only converting dose: ",dose_path," in",self.raw_dicom_folder, )
                return self.convert_dose2nii(dose,id=i)
        
        raise RuntimeError("can't even find the dicom for dose, please organize everything as instruction")

    def convert_dose2nii(self,dose_data,id="",out_path=None):
        dose=dose_data.pixel_array.astype(float)
        dose=dose*float(dose_data.DoseGridScaling)

        img=sitk.GetImageFromArray(dose)
        img.SetOrigin(dose_data.ImagePositionPatient)
        #img.SetSpacing(self.voxel_resolution)
        img.SetSpacing([dose_data.PixelSpacing[0],dose_data.PixelSpacing[1],self.voxel_resolution[2]])

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(self.get_ct())
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        image=resample.Execute(img)
        if out_path==None:
            out_path=join(self.nii_folder,"dose.nii.gz")#"+str(id)+"
        sitk.WriteImage(image,out_path)
        


    def resample_to_ct(self,image: sitk.Image,transform=sitk.Euler3DTransform(),
                 interpolator = sitk.sitkNearestNeighbor,default_value=0):

        reference_image=self.get_ct()
    
        return sitk.Resample(image, reference_image, transform,
                                interpolator, default_value) 
  
    def confirm_and_label(self):
        ct=self.get_ct()
        shape=ct.shape
        type_list=["PTV","OAR","Body"]
        for type in type_list:
            #mask=np.zeros(shape)
            # if type=="Body":
            #     img=np.array(shape).fill(1)
            # else:
            img= np.zeros(shape)
            for value in sorted(self.structure_label[type].keys()):
                #combining_mask
                mask=np.zeros(shape)
                for structure in self.structure_label[type][str(value)]:
                    mask=mask+sitk.GetArrayFromImage(sitk.ReadImage(join(self.nii_folder,structure+".nii.gz")))
                #labeling(valuing) combined mask
                mask=mask>0
                mask=mask*int(value)
                #adding labled_mask to img
                if np.sum(mask)>0:
                    if type=="OAR":
                        empty= img<=0
                        img=np.add(img,mask*empty)
                    elif type=="PTV":
                        empty= mask<=0
                        img=np.add(img*empty,mask)
                    else:#"Body shouldn't have multi structure? but the body_mask_value should be binary so it doesn't matter"
                        img=np.add(img,mask)
                else:
                    print("WARNING: Can't find any listing structure's nifti file in ",self.nii_folder, "for value", value,\
                        "please check the dataset.json or structure_name.json to make sure the name match up ")
                                       
            if type=="PTV":
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0001.nii.gz"))
            elif(type =="OAR"):
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0002.nii.gz"))
            else:
                if np.sum(img)<=0:
                    img= sitk.GetImageFromArray(img=np.array(shape).fill(1).astype(np.int32))
                    img.CopyInformation(ct)
                    sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0003.nii.gz")) 
#            shutil.copy2(join(source_path,"ct.nii.gz"),join("Dataset504","images"+str(dist_str),"0"+patient_num+"_0000.nii.gz"))    
                        
                    

        

            
    def confirm_structure_and_2nii(self):
        ''' search and "only" extracing "first" found candidate_roi_name in each defined OAR by order 
            e.g. structure_name:{"Femur_Head": ["Femur_Head_L","Femur_Head_R"]}. 
            We are "only" going to use "Femur_Head_L" if it exist, if not we would search for "Femur_Head_L . 
            If we can't find both, the mask would be left empty in this stage.
            
            If you want both "Femur_Head_L" and "Femur_Head_R", please formate like this
            structure_name:={"Femur_Head_L": ["Femur_Head_L"],
                                "Femur_Head_R": ["Femur_Head_R"]}
                                
            And if want them to combine as 1 structure, you can label them as same value, like this
            structure_label:{"OAR": {"50":["Femur_Head_L","Femur_Head_R"]}
        '''        
        ct=self.get_ct()
        structure_list= self.structure.keys() 
        
        
        for struct_file in os.listdir(self.struct_folder):
            # Load existing RT Struct. Requires the series path and existing RT Struct path
            rtstruct = RTStructBuilder.create_from(
            dicom_series_path=self.ct_folder,
            rt_struct_path=join(self.struct_folder,struct_file))
            roi_list=rtstruct.get_roi_names()

            for structure in structure_list:
                for name in self.structure_name[structure]:
                    if name in roi_list:
                        mask= sitk.GetImageFromArray( np.transpose(rtstruct.get_roi_mask_by_name(name)*int(1), axes=[2,0,1]).astype(np.int32))
                        mask.CopyInformation(ct)
                        sitk.WriteImage(mask,join(self.nii_folder,structure+".nii.gz"))
                        structure_list.remove(structure)
                        break    
            if len(structure_list)>0:
                print("WARNING: Can't find structure of these", structure_list)
class NiftiLabelr():#patient wise
    def __init__(self,dataset_name,patient_folder,patient_id,tr_ts="Tr"):      
        self.dataset_name=dataset_name 
        self.patient_folder=patient_folder
        self.patient_id=patient_id
        self.tr_ts=tr_ts
        
        self.raw_dicom_folder = join(nnUNet_dicom,self.dataset_name,tr_ts,patient_folder)
        self.raw_dataset_folder = join(nnUNet_raw,self.dataset_name)

        self.structure_label = load_json(join(nnUNet_dicom,self.dataset_name, 'structure.json'))["structure_label"]


        self.nii_folder=join(self.raw_dicom_folder,"Nifti")
        
        
  
    def get_ct(self,save=True):
        ct_path=join(self.nii_folder,"ct.nii.gz")
        if os.path.isfile(ct_path):
            return sitk.ReadImage(ct_path)
        
        if join(self.nii_folder,"img.nii.gz"):
            print('"ct.nii.gz" is not in Nifti folder but found "img.nii.gz".\n'\
                    'I am assuming img is the ct or at least the img you want to use as input\n'\
                    'so I am Renaming img.nii.gz to ct.nii.gz....\n'\
                    'if that is not what you want.... please start over and see the instruction')
            os.rename(join(self.nii_folder,"img.nii.gz"),join(self.nii_folder,"ct.nii.gz"))
            return sitk.ReadImage(ct_path)
        
        if len (os.listdir(self.ct_folder)>0):
            ct=sitk_read_dcm(self.ct_folder)
            if save:
                sitk.WriteImage(ct,ct_path)
            return ct
        
        raise RuntimeError("can't even find the dicom for ct, please organize everything as instruction")
    
    def get_dose(self,save=True):
        dose_path=join(self.nii_folder,"dose.nii.gz")
        if os.path.isfile(dose_path):
            return sitk.ReadImage(dose_path)
       
        if len(os.listdir(self.dose_folder))==1:
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                #print("converting dose",i," in",self.raw_dicom_folder)
                return self.convert_dose2nii(dose,id=i)
        if(len(os.listdir(self.dose_folder))>1):
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                print("WARNING! Multi Dose Plan Detected: only converting dose: ",dose_path," in",self.raw_dicom_folder, )
                return self.convert_dose2nii(dose,id=i)
        
        raise RuntimeError("can't even find the dicom for dose, please organize everything as instruction")

    def convert_dose2nii(self,dose_data,id="",out_path=None):
        dose=dose_data.pixel_array.astype(float)
        dose=dose*float(dose_data.DoseGridScaling)

        img=sitk.GetImageFromArray(dose)
        img.SetOrigin(dose_data.ImagePositionPatient)
        #img.SetSpacing(self.voxel_resolution)
        img.SetSpacing([dose_data.PixelSpacing[0],dose_data.PixelSpacing[1],self.voxel_resolution[2]])

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(self.get_ct())
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        image=resample.Execute(img)
        if out_path==None:
            out_path=join(self.nii_folder,"dose.nii.gz")#"+str(id)+"
        sitk.WriteImage(image,out_path)
        


    def resample_to_ct(self,image: sitk.Image,transform=sitk.Euler3DTransform(),
                 interpolator = sitk.sitkNearestNeighbor,default_value=0):

        reference_image=self.get_ct()
    
        return sitk.Resample(image, reference_image, transform,
                                interpolator, default_value) 
  
    def confirm_and_label(self):
        ct=self.get_ct()
        shape=ct.shape
        type_list=["PTV","OAR","Body"]
        for type in type_list:
            #mask=np.zeros(shape)
            # if type=="Body":
            #     img=np.array(shape).fill(1)
            # else:
            img= np.zeros(shape)
            for value in sorted(self.structure_label[type].keys()):
                #combining_mask
                mask=np.zeros(shape)
                for structure in self.structure_label[type][str(value)]:
                    mask=mask+sitk.GetArrayFromImage(sitk.ReadImage(join(self.nii_folder,structure+".nii.gz")))
                #labeling(valuing) combined mask
                mask=mask>0
                mask=mask*int(value)
                #adding labled_mask to img
                if np.sum(mask)>0:
                    if type=="OAR":
                        empty= img<=0
                        img=np.add(img,mask*empty)
                    elif type=="PTV":
                        empty= mask<=0
                        img=np.add(img*empty,mask)
                    else:#"Body shouldn't have multi structure? but the body_mask_value should be binary so it doesn't matter"
                        img=np.add(img,mask)
                else:
                    print("WARNING: Can't find any listing structure's nifti file in ",self.nii_folder, "for value", value,\
                        "please check the dataset.json or structure_name.json to make sure the name match up ")
                                       
            if type=="PTV":
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0001.nii.gz"))
            elif(type =="OAR"):
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0002.nii.gz"))
            else:
                if np.sum(img)<=0:
                    img= sitk.GetImageFromArray(img=np.array(shape).fill(1).astype(np.int32))
                    img.CopyInformation(ct)
                    sitk.WriteImage(mask,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0003.nii.gz")) 
#            shutil.copy2(join(source_path,"ct.nii.gz"),join("Dataset504","images"+str(dist_str),"0"+patient_num+"_0000.nii.gz"))    
                        
                    

        

            
    def confirm_structure_and_2nii(self):
        ''' search and "only" extracing "first" found candidate_roi_name in each defined OAR by order 
            e.g. structure_name:{"Femur_Head": ["Femur_Head_L","Femur_Head_R"]}. 
            We are "only" going to use "Femur_Head_L" if it exist, if not we would search for "Femur_Head_L . 
            If we can't find both, the mask would be left empty in this stage.
            
            If you want both "Femur_Head_L" and "Femur_Head_R", please formate like this
            structure_name:={"Femur_Head_L": ["Femur_Head_L"],
                                "Femur_Head_R": ["Femur_Head_R"]}
                                
            And if want them to combine as 1 structure, you can label them as same value, like this
            structure_label:{"OAR": {"50":["Femur_Head_L","Femur_Head_R"]}
        '''        
        ct=self.get_ct()
        structure_list= self.structure.keys() 
        
        
        for struct_file in os.listdir(self.struct_folder):
            # Load existing RT Struct. Requires the series path and existing RT Struct path
            rtstruct = RTStructBuilder.create_from(
            dicom_series_path=self.ct_folder,
            rt_struct_path=join(self.struct_folder,struct_file))
            roi_list=rtstruct.get_roi_names()

            for structure in structure_list:
                for name in self.structure_name[structure]:
                    if name in roi_list:
                        mask= sitk.GetImageFromArray( np.transpose(rtstruct.get_roi_mask_by_name(name)*int(1), axes=[2,0,1]).astype(np.int32))
                        mask.CopyInformation(ct)
                        sitk.WriteImage(mask,join(self.nii_folder,structure+".nii.gz"))
                        structure_list.remove(structure)
                        break    
            if len(structure_list)>0:
                print("WARNING: Can't find structure of these", structure_list)
