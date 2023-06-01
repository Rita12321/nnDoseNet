# %%
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import threading
from file_tool import create_folder_withsub,read_folder,check_if_list_name_match
import pydicom

# %%
MAIN_DIR="/home/changh/Documents/2023_SegmentationProject/Data"
DICOM_PATH="BackUpData/Sorted dicom"
LABEL_PATH="SAG"

#need to create
NIFTI_PATH="NIFTI_from_dcm"
RESAMPLED_FOLDER="dicom_resample256"
STRIPPED_FOLDER="dicom_resample256_stripped"


def reSample_fixedSize_fixedSpacing (img_sitk,interpolation=2, new_size=256, new_spacing=0.75):
    
    new_img = sitk.Image(new_size, new_size, new_size, sitk.sitkFloat32)
    new_img.SetSpacing([new_spacing, new_spacing, new_spacing])
    new_img.SetOrigin(img_sitk.GetOrigin())
    new_img.SetDirection(img_sitk.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(new_img)
    resampler.SetInterpolator(interpolation)
    resampler.SetDefaultPixelValue(img_sitk[0,0,0])
 

    resampled = resampler.Execute(img_sitk)

    return resampled

def reSample_to_reference_img (reference_img, img_sitk, interpolation=2,DefaultPixelValue=0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolation)

    resampled = resampler.Execute(img_sitk)

    return resampled


def sitk_dcm2nii(in_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    dicom = reader.Execute()
    return dicom

    

def threshold_based_crop(image, mask):#, resample=False
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
    

    """ 
    
    
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 1
    
    mask = sitk.BinaryImageToLabelMap(mask)
    mask = sitk.Cast(mask, sitk.sitkInt16)
    # mask_surface = sitk.LabelContour(mask == 1, True)
    
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(mask, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    
    cropped=sitk.RegionOfInterest(image,
        bounding_box[int(len(bounding_box) / 2) :],
        bounding_box[0 : int(len(bounding_box) / 2)],)
    
    sz = cropped.GetSize()

    return cropped,label_shape_filter.GetCentroid(outside_value),sz

# %%
def center_resample_mr_ct(name,save_folder=RESAMPLED_FOLDER,mask_folder=STRIPPED_FOLDER,
                          new_size=256):#, new_spacing=1,interpolation=2,DICOM=False
    
    
    ct_path=os.path.join(mask_folder,"CT_Stripped",name+".nii.gz")
    mr_path=os.path.join(mask_folder,"MR_Stripped",name+".nii.gz")
    ct_mask_path=os.path.join(mask_folder,"CT_Mask",name+".nii.gz")
    mr_mask_path=os.path.join(mask_folder,"MR_Mask",name+".nii.gz")
    
    try:    
        # if DICOM:
        #     reader = sitk.ImageSeriesReader()
        #     dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
        #     reader.SetFileNames(dicom_names)
        #     ct_img = reader.Execute()
        # else:
        ct_img = sitk.ReadImage(ct_path)
        ct_img=reSample_fixedSize_fixedSpacing (ct_img,interpolation=2, new_size=ct_img.GetSize()[0], new_spacing=0.8)
        
        
        ct_mask = sitk.ReadImage(ct_mask_path)
        ct_mask=reSample_to_reference_img(ct_img,ct_mask)
        ct_cropped,ct_center,ct_cropped_size= threshold_based_crop(ct_img,ct_mask)
        #print("CT_cropped",ct_cropped.GetSize())
        
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetConstant(ct_img[0,0,0])
        pad_filter.SetPadLowerBound((int(new_size/2 - np.floor(ct_cropped_size[0]/2)), int(new_size/2 - np.floor(ct_cropped_size[1]/2)), int(new_size/2 - np.floor(ct_cropped_size[2]/2))))
        pad_filter.SetPadUpperBound((int(new_size/2 - np.ceil(ct_cropped_size[0]/2)), int(new_size/2 - np.ceil(ct_cropped_size[1]/2)), int(new_size/2 - np.ceil(ct_cropped_size[2]/2))))
        ct_padded = pad_filter.Execute(ct_cropped)
        #print("CT_padded",ct_padded.GetSize())

        
        
        np.savez(os.path.join(save_folder,"center_np",name+".npz"), ct_center=ct_center, ct_cropped_size=ct_cropped_size)
        sitk.WriteImage(ct_padded,os.path.join(save_folder,"CT",name+".nii.gz"))
                    

    
    
        try:   
            # if DICOM:
            #     reader = sitk.ImageSeriesReader()
            #     dicom_names = reader.GetGDCMSeriesFileNames(mr_path)
            #     reader.SetFileNames(dicom_names)
            #     mr_img = reader.Execute()
            # else:
            mr_img = sitk.ReadImage(mr_path)
            mr_img=reSample_to_reference_img(ct_img,mr_img)
            
            mr_mask = sitk.ReadImage(mr_mask_path)
            mr_mask=reSample_to_reference_img(ct_img,mr_mask)
            
            mr_cropped,mr_center,mr_cropped_size= threshold_based_crop(mr_img,mr_mask)
            #print("mr_cropped",mr_cropped.GetSize())
            
            pad_filter2 = sitk.ConstantPadImageFilter()
            pad_filter2.SetConstant(0)
            pad_filter2.SetPadLowerBound((int(new_size/2 - np.floor(mr_cropped_size[0]/2)), int(new_size/2 - np.floor(mr_cropped_size[1]/2)), int(new_size/2 - np.floor(mr_cropped_size[2]/2))))
            pad_filter2.SetPadUpperBound((int(new_size/2 - np.ceil(mr_cropped_size[0]/2)), int(new_size/2 - np.ceil(mr_cropped_size[1]/2)), int(new_size/2 - np.ceil(mr_cropped_size[2]/2))))
            mr_padded = pad_filter2.Execute(mr_cropped)
            #print("MR_padded",mr_padded.GetSize())
            
            np.savez(os.path.join(save_folder,"center_np",name+".npz"), ct_center=ct_center, ct_cropped_size=ct_cropped_size,mr_center=mr_center, mr_cropped_size=mr_cropped_size)
            sitk.WriteImage(mr_padded,os.path.join(save_folder,"MR",name+".nii.gz"))
        except:
            print("MR!!",name)
    except:
        print("CT!! ",name)

def set_affine_trans_for_reg_with_regfile(dicom_registration):
    affine_transform = sitk.AffineTransform(3)
    registration_matrix = np.asarray(dicom_registration.RegistrationSequence[-1].MatrixRegistrationSequence[-1].
                                        MatrixSequence[-1].FrameOfReferenceTransformationMatrix).reshape((4, 4))
    registration_matrix = np.linalg.inv(registration_matrix)
    affine_transform.SetMatrix(registration_matrix[:3, :3].ravel())
    affine_transform.SetTranslation(registration_matrix[:3, -1])
    return affine_transform
def set_affine_trans_for_reg_with_matrix(registration_matrix):
    affine_transform = sitk.AffineTransform(3)
    if registration_matrix.shape !=(4, 4):
        registration_matrix.append([0.,0.,0.,1.])
    registration_matrix = np.linalg.inv(registration_matrix)
    affine_transform.SetMatrix(registration_matrix[:3, :3].ravel())
    affine_transform.SetTranslation(registration_matrix[:3, -1])
    return affine_transform
def reg_with_dcm(moving_image,dicom_registration=None,registration_matrix=None,fixed_image=None,min_value=0,method=sitk.sitkLinear,new_size=None,new_space=None):
    
    assert type(dicom_registration) is pydicom.dataset.FileDataset \
            or dicom_registration is None, 'Pass dicom_registration as pydicom.read_file(reg_file) or None'
            
    if dicom_registration is not None:
        affine_transform=set_affine_trans_for_reg_with_regfile(dicom_registration)
    elif registration_matrix is not None:  
        affine_transform=set_affine_trans_for_reg_with_matrix(registration_matrix)
    else:
        affine_transform = sitk.AffineTransform(3)
        
    if fixed_image==None:
        fixed_image=moving_image
        
    resampler=sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(affine_transform)
    resampler.SetInterpolator(method)
    resampler.SetDefaultPixelValue(min_value)
    if new_size!= None:
        resampler.SetSize([new_size, new_size, new_size])
    if new_space!=None:
        resampler.SetOutputSpacing([new_space,new_space,new_space])
    
    resampled =resampler.Execute(moving_image)
    
   
    #moving_resampled = sitk.Resample(moving_image, fixed_image, affine_transform, method, min_value,moving_image.GetPixelID())
    
    return resampled
