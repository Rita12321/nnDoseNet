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
        'ram_available': ram  
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
    return gamma

if __name__ == '__main__':

    import os
    import glob
    ref_folder="/home/changh/Documents/2023_DosePredcition/nnDose_1_dicom/prostate_data/Ts/"
    pred_folder="/home/changh/Documents/2023_DosePredcition/nndosev1/nnUNet_prediction/pre_dcm_520"
    save_folder="/home/changh/Documents/2023_DosePredcition/nndosev1/nnUNet_prediction/gamma_520"
    run_list=sorted(os.listdir(pred_folder))
    
    import threading

    # run_list=[]
    core=3
    def job(start,end):
        for i in range(start,end):
            print(i)
            try:
                path=run_list[i]
                name=path.split(".")[0][1:]
                print(path)
                print(name)
                print(os.path.join(ref_folder,"PROS_TEST_"+name+"_PROS_TEST_"+name+"*","Dicom","RD","*"))
                ref_path=glob.glob(os.path.join(ref_folder,"PROS_TEST_"+name+"_PROS_TEST_"+name+"*","Dicom","RD","*"))           
                gamma_analysis(reference=ref_path[0],evaluation=os.path.join(pred_folder,path),save_path=os.path.join(save_folder,"gamma_0"+name+".nii.gz"))

            except:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",i)
                continue




    threads=[]

    order=np.arange(0,len(run_list)+1,len(run_list)//core)
    order=np.append(order,len(run_list)+1//core)

    k=0
    for n in range(len(order)-1):
        
        threads.append(threading.Thread(target=job,args=(order[n],order[n+1])))
        
        threads[n].start()
        k+=1


    for b in range(len(order)-1):
        
        print("\nthread "+str(b)+" finish")
        
        threads[b].join()

    print("complete")
    # for i in run_list:
    #     path=run_list[i]
    #     name=path.split(".")[0][1:]
    #     print(path)
    #     print(name)
    #     print(os.path.join(ref_folder,"PROS_TEST_"+name+"_PROS_TEST_"+name+"*","Dicom","RD","*"))
    #     ref_path=glob.glob(os.path.join(ref_folder,"PROS_TEST_"+name+"_PROS_TEST_"+name+"*","Dicom","RD","*"))
        
    #     gamma_analysis(reference=ref_path[0],evaluation=os.path.join(pred_folder,path),save_path=os.path.join(save_folder,"gamma_0"+name+".nii.gz"))
        
    # reference_path  = "/home/changh/Documents/2023_DosePredcition/Data/ProsateData_harms/prostate_data/Ts/PROS_TEST_001_PROS_TEST_001_20240213_122216679/Dicom/RD/RD.2.25.215607266051682408066889968235317919247"
    # predict_path = "/home/changh/Documents/2023_DosePredcition/nndosev1/nnUNet_prediction/pre_dcm_510"
    # gamma=gamma_analysis(reference=reference_path,evaluation=predict_path,save_path="/home/changh/Documents/2023_DosePredcition/temp/img_gamma.nii.gz")