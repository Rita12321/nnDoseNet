import pydicom
import pymedphys
import SimpleITK as sitk
import numpy as np
import typing
def find_dose_spacing(reference):
    spacing=[reference.PixelSpacing[0],reference.PixelSpacing[1],1]
    if reference.SliceThickness is None:
        spacing[2]=spacing[0]
        return spacing
    
    if reference.SliceThickness is not None:
        # if str(reference.SliceThickness).isnumeric():
            spacing[2] = int(reference.SliceThickness)
            return spacing
    
    spacing[2]=spacing[0]

    return spacing

def gamma_analysis(reference: typing.Union[pydicom.dataset.FileDataset, str],
                   evaluation: typing.Union[pydicom.dataset.FileDataset, str],
                   mask_path,
                   save_path=None,
                   percent=3,mm=2,cutoff=20,max_gamma=1,ram=2**32):#2**32): 
    
    # just checking
    if isinstance(reference, str):
        reference = pydicom.read_file(reference, force=True)
    if isinstance(evaluation, str):
        evaluation = pydicom.read_file(evaluation, force=True)        
    
    if not isinstance(reference, pydicom.dataset.FileDataset):
        raise ValueError('Cannot get ground truth/reference dose dicom file, wrong file type.')
    
    if not isinstance(evaluation, pydicom.dataset.FileDataset):
        raise ValueError('Cannot get predict/evaluation dose dicom file, wrong file type.') 
    
    
    axes_reference, dose_reference = pymedphys.dicom.zyx_and_dose_from_dataset(reference)
    axes_evaluation, dose_evaluation = pymedphys.dicom.zyx_and_dose_from_dataset(evaluation)

    gamma_options = {
        'dose_percent_threshold': percent,
        'distance_mm_threshold': mm,
        'lower_percent_dose_cutoff': cutoff,
        'interp_fraction': 10,  # Should be 10 or more
        'max_gamma': max_gamma,
        'random_subset': None,  # Can be used to get quick pass rates
        'local_gamma': True,  # Change to false for global gamma
        'ram_available': ram,
        'quiet':True  
    }

    gamma = pymedphys.gamma(
        axes_reference, dose_reference, 
        axes_evaluation, dose_evaluation,
        **gamma_options)
    if save_path==None:
        return gamma
    gamma_np=np.nan_to_num(gamma,-1)
    image=sitk.GetImageFromArray(gamma_np)
    
    spacing=find_dose_spacing(reference=evaluation) 

    image.SetSpacing(spacing)
    image.SetDirection(list(evaluation.ImageOrientationPatient)+[0,0,1])
    image.SetOrigin(evaluation.ImagePositionPatient)
    
    sitk.WriteImage(image,save_path)
    pass_gamma,mask_num=pass_rate(max_gamma,gamma_np,image,mask_path)
    return gamma_options,pass_gamma,mask_num
    
def pass_rate(max_gamma, gamma, gamma_image, mask_path):
    #mask=np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).flatten())
    mask=sitk.ReadImage(mask_path)
    mask=resample2ref(gamma_image,mask)
    mask=sitk.GetArrayFromImage(mask).flatten()
    mask=np.nonzero(mask)[0]
    
    gamma=np.take(gamma.flatten(),mask)
    pass_gamma=len(np.nonzero(gamma<max_gamma)[0])
    return pass_gamma,len(mask)

def resample2ref(ref:sitk.Image,new:sitk.Image):

    
    # new.SetOrigin(ref.GetOrigin())
    # new.SetSpacing(ref.GetSpacing())
   

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(new)



def get_bounding_box(mask_arr):

    # assumption is that array is (x, y, z)

    x_min, x_max = np.min(np.where(np.any(mask_arr, axis=(1, 2)))), \
                   np.max(np.where(np.any(mask_arr, axis=(1, 2))))

    y_min, y_max = np.min(np.where(np.any(mask_arr, axis=(0, 2)))), \
                   np.max(np.where(np.any(mask_arr, axis=(0, 2))))

    z_min, z_max = np.min(np.where(np.any(mask_arr, axis=(0, 1)))), \
                   np.max(np.where(np.any(mask_arr, axis=(0, 1))))

    coords = {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max, 'zmin': z_min, 'zmax': z_max}

    return coords
def get_bounding_box_of_ref_and_prediction(ref,pred):
    ref=ref>1
    pred=pred>1
    mask=ref | pred
    bb_cor=get_bounding_box(mask)
    ref=ref[bb_cor['xmin']:bb_cor['xmax'],bb_cor['ymin']:bb_cor['ymax'],bb_cor['zmin']:bb_cor['zmax']]
    pred=pred[bb_cor['xmin']:bb_cor['xmax'],bb_cor['ymin']:bb_cor['ymax'],bb_cor['zmin']:bb_cor['zmax']]
    return ref, pred, bb_cor
def rewrite_dcm(reference_dicom, new_array):


    new_dicom=reference_dicom.copy()
    if reference_dicom.BitsAllocated == 16:
        out_type = 'uint16'
    elif reference_dicom.BitsAllocated == 32:
        out_type = 'uint32'
    else:
        raise ValueError(
            'Must be 16-bit or 32-bit unsigned integer data in `dicom_path` dicom'
        )
        
    # ori_dose=ori_dose.astype(float)
    ori_dose=sitk.GetImageFromArray(reference_dicom.pixel_array)
    ori_dose.SetOrigin(reference_dicom.ImagePositionPatient)
    ori_dose.SetSpacing([reference_dicom.PixelSpacing[0],reference_dicom.PixelSpacing[1],reference_dicom.PixelSpacing[0]])
    
    
    new_dicom.DoseGridScaling = float(new_array.max()) / float(np.iinfo(out_type).max)
    dose_map_scaled = new_array / new_dicom.DoseGridScaling
 

    #print(dose_map_scaled.dtype,out_type,np.min(dose_map_scaled),np.max(dose_map_scaled),dose_map_scaled.shape)
    dose_map_scaled = dose_map_scaled.astype(out_type)
    new_dicom.PixelData = dose_map_scaled.tobytes()
    return new_dicom
    

 