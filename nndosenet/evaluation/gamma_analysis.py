import pydicom
import pymedphys
import SimpleITK as sitk
import numpy as np
import typing
def find_dose_spacing(reference):
    spaceing=[reference.PixelSpacing[0],reference.PixelSpacing[1],1]
    if reference.SliceThickness is None:
        spaceing[2]=spaceing[0]
        return spaceing
    
    if int(reference.SliceThickness).isnumeric():
        spaceing[2]=reference.SliceThickness
        return spaceing
    
    spaceing[2]=spaceing[0]

    return spaceing

def gamma_analysis(reference: typing.Union[pydicom.dataset.FileDataset, str],
                   evaluation: typing.Union[pydicom.dataset.FileDataset, str],
                   mask_path,
                   save_path=None,
                   percent=3,mm=2,cutoff=20,max_gamma=1,ram=2**32): 
    
    # if isinstance(reference, str):
    #     reference = pydicom.read_file(reference, force=True)
    # if isinstance(evaluation, str):
    #     evaluation = pydicom.read_file(evaluation, force=True)        
    
    # if not isinstance(reference, pydicom.dataset.FileDataset):
    #     raise ValueError('Cannot get ground truth/reference dose dicom file, wrong file type.')
    
    # if not isinstance(evaluation, pydicom.dataset.FileDataset):
    #     raise ValueError('Cannot get predict/evaluation dose dicom file, wrong file type.') 
       
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
    # image=sitk.GetImageFromArray(gamma_np)  

    image=sitk.GetImageFromArray(gamma_np)
    
    spacing=find_dose_spacing(reference=reference) 

    image.SetSpacing(spacing)
    image.SetOrigin(reference.ImagePositionPatient)
    sitk.WriteImage(image,save_path)
    pass_gamma,mask_num=pass_rate(max_gamma,gamma_np,image,mask_path)
    return gamma_options,pass_gamma,mask_num
    
def pass_rate(max_gamma, gamma, gamma_image, mask_path):
    #mask=np.nonzero(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).flatten())
    mask=sitk.ReadImage(mask_path)
    mask=resample2ref(gamma_image,mask)
    mask=sitk.GetArrayFromImage(mask).flatten()
    mask=np.nonzero(mask)[0]
    # print(np.nonzero(ori_mask.flatten()))
    # print(np.nonzero(mask.flatten()))
    # mask=np.nonzero(.flatten())
    
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
if __name__ == '__main__':
    max_gamma=1
    gamma_image=sitk.ReadImage("/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/prostate_UAB/Dataset099_prostate_UAB/Prediction/2d/Gamma/0035.nii.gz")
    gamma=sitk.GetArrayFromImage(gamma_image)
    mask_path="/home/changh/Documents/2023_DosePredcition/nnDose_1_dicom/prostate_UAB/Ts/PROS_TEST_035_PROS_TEST_035_20240213_122201744/Nifti/Body.nii.gz"
    print(pass_rate(max_gamma, gamma, gamma_image, mask_path))
 