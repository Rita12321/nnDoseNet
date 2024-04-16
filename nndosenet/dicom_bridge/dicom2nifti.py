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
from nndosenet.paths import nnDose_raw,nnDose_dicom
from batchgenerators.utilities.file_and_folder_operations import *

def sitk_read_dcm(in_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    dicom = reader.Execute()
    return dicom 

           
class DicomConverter():#patient wise
    def __init__(self,folder,patient_folder,runn=False):      
        
        self.patient_folder=patient_folder
        
        
        self.raw_dicom_folder = join(folder,patient_folder)
        
        self.structure_name  = load_json(join(os.path.dirname(folder),'structure.json'))["structure_name"]

        
        self.ct_folder=join(self.raw_dicom_folder,"Dicom","CT")
        self.dose_folder=join(self.raw_dicom_folder,"Dicom","RD")
        self.struct_folder=join(self.raw_dicom_folder,"Dicom","RS")
        self.nii_folder=join(self.raw_dicom_folder,"Nifti")
        if runn:  
            self.run()
        
    def run(self):
        print("Converting Nifti for case ",self.patient_folder)
        print("Getting CT ")
        self.get_ct()
        print("Getting Dose map ")
        self.get_dose()
        print("Getting structure")
        self.confirm_structure_and_2nii()    
  
    def get_ct(self,save=True):
        ct_path=join(self.nii_folder,"ct.nii.gz")
        if os.path.isfile(ct_path):
            return sitk.ReadImage(ct_path)
        
        # from dicom
        if len (os.listdir(self.ct_folder))>0:
            ct=sitk_read_dcm(self.ct_folder)
            if save:
                sitk.WriteImage(ct,ct_path)
            return ct
        
        #form nifti img.nii.gz        
        if os.path.isfile(join(self.nii_folder,"img.nii.gz")):
            print('"ct.nii.gz" is not in Nifti folder but found "img.nii.gz".\n'\
                    'I am assuming "img.nii.gz" is the ct or at least the img you want to use as input\n'\
                    'so I am Renaming img.nii.gz to ct.nii.gz....\n'\
                    'if this is not what you want.... please start over and see the instruction')
            os.rename(join(self.nii_folder,"img.nii.gz"),join(self.nii_folder,"ct.nii.gz"))
            return sitk.ReadImage(ct_path)
        

        
        raise RuntimeError("If you are using Nifti, please name ct in ct.nii.gz in Nifti folder"\
                            "If using dicom please start with dicom's process, and remember to name CT slices startwith CT. "\
                            "Please read the instruction, thank u")
    
    def get_dose(self):
        dose_path=join(self.nii_folder,"dose.nii.gz")
        if os.path.isfile(dose_path):
            return sitk.ReadImage(dose_path)
       
        if len(os.listdir(self.dose_folder))==1:
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                return self.convert_dose2nii(dose,out_path=join(self.nii_folder,"dose.nii.gz"))
            
        if(len(os.listdir(self.dose_folder))>1):
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose=pydicom.dcmread(join(self.dose_folder,dose_path))
                print("WARNING! Multi Dose Plan Detected: only converting dose: ",dose_path," in",self.raw_dicom_folder, )
                return self.convert_dose2nii(dose,out_path=join(self.nii_folder,"dose.nii.gz"))
        
        raise RuntimeError("can't even find the dicom for dose, please organize everything as instruction")

    def convert_dose2nii(self,dose_data,out_path=None):
        if out_path==None:
            out_path=join(self.nii_folder,"dose.nii.gz")
            
        ct=self.get_ct()
        
        #Todo:try except using PixelData frombuffer
        
        
        
        dose=dose_data.pixel_array.astype(float)
        dose=dose*float(dose_data.DoseGridScaling)
        
        dose=sitk.GetImageFromArray(dose)
        dose.SetOrigin(dose_data.ImagePositionPatient)
        dose.SetSpacing([dose_data.PixelSpacing[0],dose_data.PixelSpacing[1],ct.GetSpacing()[2]])

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(ct)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        dose_out=resample.Execute(dose)
        sitk.WriteImage(dose_out,out_path)

    def resample_to_ct(self,image: sitk.Image,transform=sitk.Euler3DTransform(),
                 interpolator = sitk.sitkNearestNeighbor,default_value=0):
        reference_image=self.get_ct()
        return sitk.Resample(image, reference_image, transform,
                                interpolator, default_value) 
                    
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
        structure_list= list(self.structure_name.keys())
        structure_list_found=[]
        
        for struct_file in os.listdir(self.struct_folder):
            # Load existing RT Struct. Requires the series path and existing RT Struct path
            rtstruct = RTStructBuilder.create_from(
            dicom_series_path=self.ct_folder,
            rt_struct_path=join(self.struct_folder,struct_file))
            roi_list=rtstruct.get_roi_names()
            
            for structure in structure_list:
                if structure in structure_list_found:
                    continue
                for name in self.structure_name[structure]:
                    #print(name)
                    if name in roi_list:
                        #print(name)
                        mask= sitk.GetImageFromArray( np.transpose(rtstruct.get_roi_mask_by_name(name)*int(1), axes=[2,0,1]).astype(np.int32))
                        mask.CopyInformation(ct)
                        sitk.WriteImage(mask,join(self.nii_folder,structure+".nii.gz"))
                        structure_list_found.append(structure)
                        break 
                      
        if len(structure_list)>len(structure_list_found):
            not_found=set(structure_list)-set(structure_list_found)#NiftiLabeler
            print("WARNING: Can't find structure of these", not_found)
            print("These are the list roi_list")
            print(roi_list)
    

class NiftiLabeler(DicomConverter):
    
    def __init__(self,folder,patient_folder,patient_id,dataset_name,dataset_id,tr_ts="Tr",runn=False):  
           
        self.dataset_name=dataset_name 
        self.patient_folder=patient_folder
        self.patient_id=str(patient_id).zfill(4)
        print(self.patient_id)
        self.tr_ts=tr_ts

        
        self.raw_dicom_folder = join(folder,patient_folder)
        self.raw_dataset_folder = join(nnDose_raw,self.dataset_name)
        if not os.path.isdir(self.raw_dataset_folder):
            os.makedirs(self.raw_dataset_folder)
            print("Create folder in nnDose_raw: ",self.raw_dataset_folder)

        self.structure_label = load_json(join(os.path.dirname(folder),'structure.json'))["structure_label"]  
        
        self.nii_folder=join(self.raw_dicom_folder,"Nifti")
        if runn:
            self.run()
    def run(self):
        self.confirm_and_label()   
    def check_label_sanity():
        #TODO: check label sanity
        pass
    def confirm_and_label(self):
        ct=self.get_ct()
        shape=sitk.GetArrayFromImage(ct).shape
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
                    if os.path.isfile(join(self.nii_folder,structure+".nii.gz")):
                        mask=mask+sitk.GetArrayFromImage(sitk.ReadImage(join(self.nii_folder,structure+".nii.gz")))
                    # else:
                    #     print("Warning: Cannot find",structure,"in",self.patient_folder,"!!Please double check")
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
                        "in",type,"the structure of ",self.structure_label[type][str(value)],\
                        "please check the dataset.json or structure_name.json to make sure the name match up ")
                                       
            if type=="PTV":
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(img,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0001.nii.gz"))
            elif(type =="OAR"):
                img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(img,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0002.nii.gz"))
            else:
                if np.sum(img)<=0:
                    img= sitk.GetImageFromArray(img=np.array(shape).fill(1).astype(np.int32))
                else:
                    img= sitk.GetImageFromArray(img.astype(np.int32))
                img.CopyInformation(ct)
                sitk.WriteImage(img,join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0003.nii.gz")) 
        
        #Copy dose and ct
        shutil.copy2(join(self.nii_folder,"ct.nii.gz"),join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0000.nii.gz")) 
        shutil.copy2(join(self.nii_folder,"dose.nii.gz"),join(self.raw_dataset_folder,"labels"+self.tr_ts,self.patient_id+".nii.gz")) 
#            shutil.copy2(join(source_path,"ct.nii.gz"),join("Dataset504","images"+str(dist_str),"0"+patient_num+"_0000.nii.gz"))    


                