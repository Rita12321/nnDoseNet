
import multiprocessing as mp
import os
import typing as typ

import numpy as np
import pydicom
import SimpleITK as sitk



def resample2gt(ref_dicom:sitk.Image,prediction:sitk.Image):
    #dose=ref_dicom.pixel_array.astype(float)

    
    # prediction.SetOrigin(ref_dicom.GetOrigin())
    # prediction.SetSpacing(ref_dicom.GetSpacing())
   

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref_dicom)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(prediction)

def dose_to_dicom(predict_nifti_path: str,reference_dicom_path: str,save_path: str,for_gamma=True):
    """
    Copyright (C) 2020-2024 M. Gronberg and S. Gay
    #TODO: include their info maybe?
    Args:
        dose:       One of a Numpy array, SimpleITK image, or string or
                    path-like object to the saved dose file. This is the dose
                    map to converto to a dicom

        ref_dicom:  One of a PyDicom loaded object, or string or path-like
                    object, the path to an existing RD*.dcm
                    for the patient from which to copy information

        save_path:  String or path-like object, the path to write the new
                    dicom
        
        for_gamma: Return  pydicom.dataset.FileDataset format of reference and predict dose or not

    """
    reference_dicom = pydicom.dcmread(reference_dicom_path)
    predict_dicom = reference_dicom.copy()

    dose_sitk=sitk.ReadImage(predict_nifti_path)
    spacing_z=dose_sitk.GetSpacing()[2]
    pre_np_dose = sitk.GetArrayFromImage(dose_sitk)

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

    pre_np_dose = pre_np_dose.astype(float)
    predict_dicom.DoseGridScaling = float(pre_np_dose.max()-1) / float(np.iinfo(out_type).max)
    dose_map_scaled = pre_np_dose / predict_dicom.DoseGridScaling
 

    # ori_dose=ori_dose.astype(float)
    ori_dose=sitk.GetImageFromArray(reference_dicom.pixel_array)
    ori_dose.SetOrigin(reference_dicom.ImagePositionPatient)
    ori_dose.SetSpacing([reference_dicom.PixelSpacing[0],reference_dicom.PixelSpacing[1],reference_dicom.PixelSpacing[0]])
    
    pre_dose=sitk.GetImageFromArray(dose_map_scaled)
    pre_dose.SetOrigin(dose_sitk.GetOrigin())
    pre_dose.SetSpacing(dose_sitk.GetSpacing())

    dose_map=sitk.GetArrayFromImage(resample2gt(ori_dose,pre_dose))
    #print(type(dose_map.dtype))
    dose_map = dose_map.astype(out_type)
    
    predict_dicom.PixelData = dose_map.tobytes()
    

    # Change Series number       
    predict_dicom[0x0020, 0x0011].value=str(reference_dicom[0x0020, 0x0011].value)+"0"
    
    # Change ReferencedSOPInstanceUID (RD)
    predict_dicom[0x0008, 0x0018].value=str(reference_dicom[0x0008, 0x0018].value)+"0000"
               
    # Change ReferencedSOPInstanceUID (RP)
    predict_dicom[0x300C, 0x0002][0][0x0008, 0x1155].value=str(reference_dicom[0x300C, 0x0002][0][0x0008, 0x1155].value)+"0000"

    
    predict_dicom.save_as(save_path)
    if for_gamma:
        return reference_dicom, predict_dicom
        
