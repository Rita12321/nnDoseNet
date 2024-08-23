
import multiprocessing as mp
import os
import typing as typ

import numpy as np
import pydicom
import SimpleITK as sitk



def resample2gt(ref_dicom:sitk.Image,prediction:sitk.Image):
    # dose = ref_dicom.pixel_array.astype(float)
    # prediction.SetOrigin(ref_dicom.GetOrigin())
    # prediction.SetSpacing(ref_dicom.GetSpacing())
    
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref_dicom)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(prediction)

def open_kbp_trouble(nifti_img):
    nifti_data = sitk.GetArrayFromImage(nifti_img)
    nifti_data = np.transpose(nifti_data, (2, 0, 1))
    nifti_data = np.flip(nifti_data, axis=0)
    newimage = sitk.GetImageFromArray(nifti_data)
    newimage.CopyInformation(nifti_img)
    newimage.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, -1])
    newimage.SetOrigin([newimage.GetOrigin()[0],newimage.GetOrigin()[1],newimage.GetOrigin()[2]+(nifti_data.shape[2]-1)*newimage.GetSpacing()[2] ])
    return newimage, sitk.GetArrayFromImage(newimage)

def dose_to_dicom(predict_nifti_path: str,reference_dicom_path: str,save_path: str,for_gamma=True):
    """
    Copyright (C) 2020-2024 M. Gronberg and S. Gay
    Args:
        dose:       One of a Numpy array, SimpleITK image, or string or
                    path-like object to the saved dose file. This is the dose
                    map to convert to a dicom

        ref_dicom:  One of a PyDicom loaded object, or string or path-like
                    object, the path to an existing RD*.dcm
                    for the patient from which to copy information

        save_path:  String or path-like object, the path to write the new
                    dicom
        
        for_gamma: Return  pydicom.dataset.FileDataset format of reference and predict dose or not

    """
    openkbp = True
    reference_dicom = pydicom.dcmread(reference_dicom_path)
    predict_dicom = reference_dicom.copy()

    if openkbp:
        dose_sitk,pre_np_dose = open_kbp_trouble(sitk.ReadImage(predict_nifti_path))
    else:
        dose_sitk = sitk.ReadImage(predict_nifti_path)
        # spacing_z=dose_sitk.GetSpacing()[2]
        pre_np_dose = sitk.GetArrayFromImage(dose_sitk)
    # dose_sitk,pre_np_dose = open_kbp_trouble(sitk.ReadImage(predict_nifti_path))
    
    # elif isinstance(dose, sitk.Image):
    #     spacing_z=dose.GetSpacing()[2]
    #     dose = sitk.GetArrayFromImage(dose)

    # 16 or 32-bit unsigned integers are possible
    if reference_dicom.BitsAllocated == 16:
        out_type = 'uint16'
    elif reference_dicom.BitsAllocated == 32:
        out_type = 'uint32'
    else:
        raise ValueError(
            'Must be 16-bit or 32-bit unsigned integer data in `dicom_path` dicom'
        )
        
    # ori_dose=ori_dose.astype(float)
    ori_dose = sitk.GetImageFromArray(reference_dicom.pixel_array)
    ori_dose.SetOrigin(reference_dicom.ImagePositionPatient)
    ori_dose.SetSpacing([reference_dicom.PixelSpacing[0],reference_dicom.PixelSpacing[1],reference_dicom.PixelSpacing[0]])
    
    pre_np_dose = pre_np_dose.astype(float)
    pre_np_dose = sitk.GetImageFromArray(pre_np_dose)
    pre_np_dose.SetOrigin(dose_sitk.GetOrigin())
    pre_np_dose.SetSpacing(dose_sitk.GetSpacing())
    if not openkbp:
        pre_np_dose = resample2gt(ori_dose,pre_np_dose)
    pre_np_dose = sitk.GetArrayFromImage(pre_np_dose)
    
    predict_dicom.DoseGridScaling = float(pre_np_dose.max()) / float(np.iinfo(out_type).max)
    dose_map_scaled = pre_np_dose / predict_dicom.DoseGridScaling
 

    #print(dose_map_scaled.dtype,out_type,np.min(dose_map_scaled),np.max(dose_map_scaled),dose_map_scaled.shape)
    dose_map_scaled = dose_map_scaled.astype(out_type)
    
        
    
    predict_dicom.PixelData = dose_map_scaled.tobytes()
    

    # Change Series number       
    predict_dicom[0x0020, 0x0011].value = str(reference_dicom[0x0020, 0x0011].value)+"0"
    
    # Change ReferencedSOPInstanceUID (RD)
    predict_dicom[0x0008, 0x0018].value = str(reference_dicom[0x0008, 0x0018].value)+"0000"
               
    # Change ReferencedSOPInstanceUID (RP)
    predict_dicom[0x300C, 0x0002][0][0x0008, 0x1155].value = str(reference_dicom[0x300C, 0x0002][0][0x0008, 0x1155].value)+"0000"

    
    predict_dicom.save_as(save_path)
    if for_gamma:
        return reference_dicom, predict_dicom
        
