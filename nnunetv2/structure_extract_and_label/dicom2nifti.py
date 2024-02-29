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
        if not os.path.isdir(os.path.join(path,sub_folder)):
            os.makedirs(os.path.join(path,sub_folder))
        elif overwrite:
            shutil.rmtree(os.path.join(path,sub_folder))
            os.makedirs(os.path.join(path,sub_folder))
            
def organize_dicom(DATA_FOLDER):
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
        -patient 1
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
                
        -patient 2
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
        -patient 3 
        ...
    """

    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        print(patient_folder)
        patient_path=os.path.join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"],overwrite=True)
        create_folderlist(patient_path,folder_list=["Nifti","Dicom"])
        type_list=["CT","RD","RS","extra"]
        create_folderlist(os.path.join(patient_path,"Dicom"),folder_list=type_list)
        for type in type_list:
            dcms=glob.glob(os.path.join(patient_path,type+"*"))
            for dcm in dcms:
                shutil.move(dcm,os.path.join(patient_path,"Dicom",type))
            if len(dcms<=0) and type!="extra": 
                    print('WARNING: cannot find the .dcm file contain" '+type+'" '\
                        'please check the if the folder directory is correct or it has" '+type+' " in file name.')             
            
        print(' finish organizing the the data in dir: '+DATA_FOLDER)
               
class DicomConverter():
    def __init__(self,patient_id=None):
        self.plans=[]        
        self.patient_id=patient_id
        
        self.raw_dataset_folder = join(nnUNet_dicom, self.dataset_name)
        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.structure_json= load_json(join(self.raw_dataset_folder, 'sructure.json'))
        
        self.ct_folder=os.path.join(self.raw_dataset_folder,"Dicom","CT")
        self.dose_folder=os.path.join(self.raw_dataset_folder,"Dicom","RD")
        #self.plan_folder=os.path.join(self.raw_dataset_folder,"Dicom","RP")
        self.struct_folder=os.path.join(self.raw_dataset_folder,"Dicom","RS")
        organize_dicom(self.raw_dataset_folder)
        self.get_patient_info_from_ct()
        
    def get_patient_info_from_ct(self):
        cts=os.listdir(self.ct_folder)
        ct=pydicom.dcmread(os.path.join(self.ct_folder,cts[0]))

        self.patient_position=ct.ImagePositionPatient
        self.patient_orientation=ct.ImageOrientationPatient
        
        ct=self.get_ct(save=True)
        self.origin=ct.GetOrigin()
        self.voxel_resolution=ct.GetSpacing()
        self.array_size=ct.GetSize()
        self.direction=ct.GetDirection()
        
  
    def get_ct(self,save=True):
        ct_path=os.path.join(self.raw_dataset_folder,"Nifti","ct.nii.gz")
        
        if not os.path.isfile(ct_path):
            if not os.path.join(self.raw_dataset_folder,"Nifti","img.nii.gz"):
                if len (os.listdir(self.ct_folder)>0):
                    ct=sitk_read_dcm(self.ct_folder)
                    if save:
                        sitk.WriteImage(ct,ct_path)
                    return ct
                else:
                    print("please organize everything in instruction...")
                    #FIXME: add raiseerror
                    return None
                  
            
        return sitk.ReadImage(ct_path)
    
    def checking_dose_in_dicom_then_convert (self,dtype=sitk.sitkUInt64):
        
        for i,dose_path in enumerate(os.listdir(self.dose_folder)):
            dose=pydicom.dcmread(os.path.join(self.dose_folder,dose_path))
            print("converting dose",i," in",self.raw_dataset_folder)
            self.convert_dose2nii(dose,id=i)

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
            out_path=os.path.join(self.raw_dataset_folder,"Nifti","dose.nii.gz")#"+str(id)+"
        sitk.WriteImage(image,out_path)


    def resample_to_ct(self,image: sitk.Image,transform=sitk.Euler3DTransform(),
                 interpolator = sitk.sitkNearestNeighbor,default_value=0):

        reference_image=self.get_ct()
    
        return sitk.Resample(image, reference_image, transform,
                                interpolator, default_value) 
        
    def resample_template_to_iso(self, 
                  image: sitk.Image,
                  origin=None,
                  transform=sitk.Euler3DTransform
                  ):
        
        image.SetOrigin(origin)
        image.SetDirection(self.direction)

        resample = sitk.ResampleImageFilter()
        #resample.SetTransform(transform)
        resample.SetReferenceImage(image)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    
        return resample.Execute(image)

    def convert_structure(self):
        ct=self.get_ct()
        ct_shape=sitk.GetArrayFromImage(ct).shape
        shape=[ct_shape[1],ct_shape[2],ct_shape[0]] #orientation different pydiocm and sitk... or np I forgot... 
        PTV= np.zeros(shape)
        OAR= np.zeros(shape)
        DoseMask= np.zeros(shape)


        dcm_dir = os.path.join(self.raw_dataset_folder,'CT')
        rs_file = os.listdir(os.path.join(self.raw_dataset_folder,'RS'))
   

        for struct_file in rs_file:

            # Load existing RT Struct. Requires the series path and existing RT Struct path
            rtstruct = RTStructBuilder.create_from(
            dicom_series_path=dcm_dir,
            rt_struct_path=os.path.join(self.raw_dataset_folder,'RS',struct_file)
            )

            # View all of the ROI names from within the image
            
            #FIXME:
            mask_list=rtstruct.get_roi_names()

            for channel in self.structure_label:

                for structure in channel:
                    mask=np.zeros(shape)
                    for name in rois[channel]:
                        if name in mask_list:
                            print(name)
                            mask=rtstruct.get_roi_mask_by_name(name)
                            mask=int(label)*mask
                            break    
                    if channel=='PTV':
                        PTV=mask
                    elif channel=="Body":
                        DoseMask=mask
                    else:
                        if OAR.shape==mask.shape:
                            empty= OAR<=0
                            OAR=np.add(OAR,mask*empty)
                        else:
                            print("shape !!!!!!!!!!!!!!!!!!", OAR.shape,mask.shape)

                        
            #print(np.unique(mask.flatten()))
        
            
        # ct=sitk.ReadImage(os.path.join(save_dir,"ct.nii.gz"))   
        # print(sitk.GetArrayFromImage(ct).shape,OAR.shape)   
        # print(sitk.GetArrayFromImage(ct).dtype,OAR.dtype)   
            
        OAR= sitk.GetImageFromArray( np.transpose(OAR, axes=[2,0,1]).astype(np.int32))
        OAR.CopyInformation(ct)
        sitk.WriteImage(OAR,os.path.join(save_dir,"OAR.nii.gz"))
        
        PTV= sitk.GetImageFromArray(np.transpose(PTV, axes=[2, 0, 1]).astype(np.int32))
        PTV.CopyInformation(ct)
        sitk.WriteImage(PTV,os.path.join(save_dir,"PTV.nii.gz"))
        
        if np.count_nonzero(DoseMask)<=0:
            DoseMask=np.full(shape,1)
        DoseMask= sitk.GetImageFromArray(np.transpose(DoseMask, axes=[2, 0, 1]).astype(np.int32))
        DoseMask.CopyInformation(ct)
        sitk.WriteImage(DoseMask,os.path.join(save_dir,"DoseMask.nii.gz"))                        

    def combine_masks(self):
        pass
        # get PTVs and OARs mask comined seperately  