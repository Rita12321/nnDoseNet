import pydicom
import pymedphys
import SimpleITK as sitk
import numpy as np

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

def gamma_analysis(refernce_path,predict_path,save_path=None,percent=3,mm=2,cutoff=20,max_gamma=2,ram=2**32):
    reference = pydicom.read_file(refernce_path, force=True)
    evaluation = pydicom.read_file(predict_path, force=True)
    
    axes_reference, dose_reference = pymedphys.dicom.zyx_and_dose_from_dataset(reference)
    axes_evaluation, dose_evaluation = pymedphys.dicom.zyx_and_dose_from_dataset(evaluation)

    gamma_options = {
        'dose_percent_threshold': percent,  # This is a bit overkill here at 1%/1mm
        'distance_mm_threshold': mm,
        'lower_percent_dose_cutoff': cutoff,
        'interp_fraction': 10,  # Should be 10 or more, see the paper referenced above
        'max_gamma': max_gamma,
        'random_subset': None,  # Can be used to get quick pass rates
        'local_gamma': True,  # Change to false for global gamma
        'ram_available': ram  # 1/2 GB
    }

    gamma = pymedphys.gamma(
        axes_reference, dose_reference, 
        axes_evaluation, dose_evaluation, 
        **gamma_options)
    if save_path==None:
        return gamma
    gamma_np=np.nan_to_num(gamma,-1)
    # image=sitk.GetImageFromArray(gamma_np)  
    print(gamma_np.shape)
    image=sitk.GetImageFromArray(gamma_np)  
    
    spacing=find_dose_spacing(reference=reference) 
    print(spacing)
    image.SetSpacing(spacing)
    image.SetOrigin(reference.ImagePositionPatient)
    sitk.WriteImage(image,save_path)
    return gamma

if __name__ == '__main__':
    reference_path  = "/home/changh/Documents/2023_DosePredcition/Data/ProsateData_harms/prostate_data/Ts/PROS_TEST_001_PROS_TEST_001_20240213_122216679/Dicom/RD/RD.2.25.215607266051682408066889968235317919247"
    predict_path = "/home/changh/Documents/2023_DosePredcition/nndosev1/nnUNet_prediction/pre_dcm_504/0001.dcm"
    gamma=gamma_analysis(refernce_path=reference_path,predict_path=predict_path,save_path="/home/changh/Documents/2023_DosePredcition/temp")