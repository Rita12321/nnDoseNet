import SimpleITK as sitk 
from rt_utils import RTStructBuilder
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid#, CTImageStorage, ExplicitVRBigEndian, ImplicitVRLittleEndian, ExplicitVRLittleEndian,RTStructureSetStorage
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import nibabel as nib

RTRS_Color_Table={
    "PTV70":[128,0,128],
    "PTV_high":[128,0,128],
    "PTV63":[153,50,204],
    "PTV_mid":[153,50,204],
    "PTV56":[128,0,128],
    "PTV_low":[128,0,128],
    "Mandible":[0,255,0],
    "Brainstem":[255,128,128],
    "Esophagus":[0,128,128],
    "Larynx":[255,128,128],
    "RightParotid":[255,255,0],
    "LeftParotid":[255,255,0],
    "SpinalCord":[0,255,255],
    "PTV":[128,0,128],
    "High_PTV":[128,0,128],
    "Bowel":[0,128,128],
    "Femur_Head_L":[255,255,0],
    "Femur_Head_R":[255,255,0],
    "Bladder":[255,128,128],
    "Rectum":[0,255,255]
}

class Dicom2Nifti_RT():
    def __init__(self, output_diocm_folder = None , img_nii_path = None,struct_nii_folder = None, dose_nii_path = None, patient_id="404", patient_name="Jane^Dose", openkbp=False, RD_TEMPLATE_PATH=None,CT_TEMPLATE_PATH=None):
        
        self.RTDOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
        if RD_TEMPLATE_PATH is None:
            RD_TEMPLATE_PATH = "RD_template.dcm"
        if CT_TEMPLATE_PATH is None:
            CT_TEMPLATE_PATH = "CT_template.dcm"
        
        self.RDDS_template = pydicom.dcmread(RD_TEMPLATE_PATH)
        self.CTDS_template = pydicom.dcmread(CT_TEMPLATE_PATH)
        
        self.openkbp = openkbp
        self.patient_name = patient_name
        self.patient_id = str(patient_id).zfill(10)
        self.Date = str(datetime.date.today()).replace('-', '')
        self.Time = str(datetime.datetime.now().time()).replace(':', '').split('.')[0]
        self.StudyInstanceUID = generate_uid()
        self.FrameOfReferenceUID = generate_uid()
        self.rtct_uid=None
        
        
        self.output_diocm_folder = os.path.join(output_diocm_folder,str(patient_id).zfill(5))
        if not os.path.exists(self.output_diocm_folder):
            os.makedirs(self.output_diocm_folder)
            
        self.CT_output_folder = os.path.join(self.output_diocm_folder,"CT")
        if not os.path.exists(self.CT_output_folder):
            os.makedirs(self.CT_output_folder)
        
        self.nifti_to_dicom_CT(self.CTDS_template,img_nii_path, self.CT_output_folder )
        self.create_dicom_rt_struct(self.CT_output_folder,struct_nii_folder, os.path.join(self.output_diocm_folder,"RS."+self.patient_id+".dcm"))
        #TODO: overwrite uid
        rtrs_uid=pydicom.dcmread(os.path.join(self.output_diocm_folder,"RS."+self.patient_id+".dcm")).SOPInstanceUID
        self.create_dicom_rt_dose(self.RDDS_template,dose_nii_path, os.path.join(self.output_diocm_folder,"RD."+self.patient_id+".dcm"),rtrs_uid)
   
    def open_kbp_trouble(self,nifti_img):
        nifti_data = sitk.GetArrayFromImage(nifti_img)
        nifti_data = np.transpose(nifti_data, (2, 0, 1))
        nifti_data = np.flip(nifti_data, axis=0)
        newimage=sitk.GetImageFromArray(nifti_data)
        newimage.CopyInformation(nifti_img)
        newimage.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, -1])
        return newimage, sitk.GetArrayFromImage(newimage)
        print(newimage.GetOrigin())
        sitk.WriteImage(newimage,"/home/changh/Documents/2023_DosePredcition/test_code/test_nifti2newdiocm/test_nifti_kbp.nii.gz")
        nifti_img = sitk.ReadImage("/home/changh/Documents/2023_DosePredcition/test_code/test_nifti2newdiocm/test_nifti_kbp.nii.gz")
        nifti_data = sitk.GetArrayFromImage(nifti_img)
        return nifti_img, nifti_data
    
  
    def nifti_to_dicom_CT(self, ds_template, nifti_file,output_folder):


  
        # nifti_img = sitk.ReadImage(nifti_file)
        # nifti_data = np.moveaxis(sitk.GetArrayFromImage(nifti_img), 0, -1)
        
        # pixel_spacing = nifti_img.GetSpacing()[:2]
        # slice_thickness = nifti_img.GetSpacing()[2]      
        
        # image_orientation_patient =list(nifti_img.GetDirection())
        # image_position_patient = nifti_img.GetOrigin()
        # print(pixel_spacing,slice_thickness, nifti_data.shape)
        # print(image_orientation_patient, image_position_patient)
        # if self.openkbp:


        #     image_orientation_patient=[1,0,0,0,0,-1,0,1,0]#[0,1,0,0,0,-1,1,0,0]#

            
        # SeriesInstanceUID= generate_uid()        

        nifti_img = sitk.ReadImage(nifti_file)
        nifti_data = sitk.GetArrayFromImage(nifti_img)

        if self.openkbp:
            nifti_img, nifti_data=self.open_kbp_trouble(nifti_img)
            #nifti_data = np.moveaxis(nifti_data, 0, 2)
            nifti_img.SetOrigin([nifti_img.GetOrigin()[0],nifti_img.GetOrigin()[1],nifti_img.GetOrigin()[2]+(nifti_data.shape[2]-1)*nifti_img.GetSpacing()[2] ])
            nifti_data = np.flip(nifti_data, axis=0)
            
        nifti_data = np.moveaxis(nifti_data, 0, 2)
        
        pixel_spacing = [nifti_img.GetSpacing()[0], nifti_img.GetSpacing()[1]]
        slice_thickness = nifti_img.GetSpacing()[2]      
        
        image_orientation_patient =list(nifti_img.GetDirection())
        image_position_patient = nifti_img.GetOrigin()
        # print(pixel_spacing, slice_thickness, nifti_data.shape)
        # print(image_orientation_patient, image_position_patient)
            
        SeriesInstanceUID= generate_uid()   
                        
        # intercept = np.floor(np.min(nifti_data))
        # slope = np.iinfo(np.uint16).max//(np.ceil(np.max(nifti_data))-intercept)
        intercept = np.min(nifti_data).astype(np.int16)
        slope = (np.max(nifti_data)-intercept)/np.iinfo(np.int16).max
        
        # Loop through each slice in the NIfTI file
        for i in range(nifti_data.shape[2]): # nifti_data.shape[1]
            # Extract a single slice
            slice_data = ((nifti_data[:,:,i]-intercept)/slope).astype(np.int16)
         
            
            # Create a new DICOM dataset
            file_meta = pydicom.dataset.FileMetaDataset()
            ds = ds_template.copy()#FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # Fill in some necessary values
            ds.PatientName = self.patient_name
            ds.PatientID = self.patient_id
            ds.Modality = "CT"
            ds.SOPInstanceUID = generate_uid()
            if i == 0:
                self.rtct_uid = ds.SOPInstanceUID 
            ds.SeriesInstanceUID = SeriesInstanceUID
            ds.StudyInstanceUID = self.StudyInstanceUID
            ds.ContentDate = self.Date
            ds.ContentTime = self.Time
            ds.StudyDate = self.Date
            ds.StudyTime = self.Time
            ds.StudyID = self.patient_id
            # Set the pixel data
            ds.Rows, ds.Columns = slice_data.shape
            ds.SamplesPerPixel = 1
            # ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 1  # 0 for unsigned integers
            ds.HighBit = 15
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.RescaleSlope = slope
            ds.RescaleIntercept = intercept
            ds.PixelData = slice_data.tobytes()
            
            # Add NIfTI metadata to DICOM
            ds.SliceLocation = float(i * slice_thickness*image_orientation_patient[-1])
            ds.ImagePositionPatient = [
                float(image_position_patient[0]),
                float(image_position_patient[1]),
                float(image_position_patient[2]+ i * slice_thickness*image_orientation_patient[-1])
            ]
  
            ds.ImageOrientationPatient = [
            float(image_orientation_patient[0]), float(image_orientation_patient[1]), float(image_orientation_patient[2]),
            float(image_orientation_patient[3]), float(image_orientation_patient[4]), float(image_orientation_patient[5])
            ]
            ds.PixelSpacing = [float(pixel_spacing[0]),float(pixel_spacing[1]) ]
            ds.SliceThickness = float(slice_thickness)#float(pixel_spacing[0])
            ds.FrameOfReferenceUID = self.FrameOfReferenceUID
            # File meta information
            # file_meta.MediaStorageSOPClassUID = CTImageStorage
            # ds.SOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            # file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()

            # Save the DICOM file
            output_file = os.path.join(output_folder, f"slice_{i:03}.dcm")
            ds.save_as(output_file)


    def load_contours(self, contour_nii_dir):
        contours = {}
        PTVs={}
        shape=None
        for file in os.listdir(contour_nii_dir):
            if file.endswith('.nii.gz'):
                if ("ct" in str(file)) or ("CT" in str(file)) or ("dose.nii.gz" in str(file)) or ("DOSE.nii.gz" in str(file)) or ("Dose.nii.gz" in str(file))  :
                    continue
                contour_name = os.path.splitext(os.path.splitext(file)[0])[0]
                contour_path = os.path.join(contour_nii_dir, file)
                contour_image = sitk.ReadImage(contour_path)
                
                if self.openkbp:
                    contour_image, contour_arr=self.open_kbp_trouble(contour_image)
                    
                contour_arr = sitk.GetArrayFromImage(contour_image)
                contour_arr = np.moveaxis(contour_arr, 0, -1)
                
                if self.openkbp and("PTV" in contour_name):
                    PTVs[contour_name] = contour_arr.astype(bool)
                    continue
                    
                contours[contour_name] = contour_arr.astype(bool)#np.moveaxis(, 0, -1)
                shape=contour_arr.shape
        temp_ptv=np.zeros(shape, dtype=bool)
        for ptv_pair in sorted(PTVs.items(),reverse=True):
            temp_ptv = temp_ptv | ptv_pair[1]
            contours[ptv_pair[0]] = temp_ptv
            

                
        
        return contours
    
    # Create new RT Struct. Requires the DICOM series path for the RT Struct.
    def create_dicom_rt_struct(self,output_CT_folder, contour_nii_dir,output_path):
        rtstruct = RTStructBuilder.create_new(dicom_series_path=output_CT_folder)
        contours = self.load_contours(contour_nii_dir)

        for i, (contour_name, contour_image) in enumerate(contours.items()):
            if contour_name in RTRS_Color_Table.keys():
                contour_color = RTRS_Color_Table[contour_name]
            else:
                contour_color = [0,100,255]
            # Add another ROI, this time setting the color, description, and name
            rtstruct.add_roi(
            mask=contour_image, 
            color=contour_color, 
            name=contour_name,
            approximate_contours=False
            )
            

        rtstruct.save(output_path)


    def create_dicom_rt_dose(self, ds, nii_file_path, output_path,rtrs_uid = None, rtrp_uid = None):
        # Read the NIfTI file
        nii_image = sitk.ReadImage(nii_file_path)
        nii_array = sitk.GetArrayFromImage(nii_image)
        if self.openkbp:
            nii_image, nii_array=self.open_kbp_trouble(nii_image)
        # Get image properties
        # nii_array = np.moveaxis(nii_array, 0, -1)
        spacing = nii_image.GetSpacing()
        origin = nii_image.GetOrigin()
        direction = nii_image.GetDirection()

        # Create the DICOM RT Dose dataset
        # file_meta = pydicom.Dataset()
        # file_meta.MediaStorageSOPClassUID = RTDOSE_STORAGE
        # file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        # file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        # file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # ds = FileDataset(output_dicom_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # # Set patient information
        ds.PatientName = self. patient_name
        ds.PatientID = self.patient_id

        # # Set general study information
        ds.StudyInstanceUID = self.StudyInstanceUID
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()

        ds.Manufacturer = "Python SimpleITK"
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SeriesDate = ds.StudyDate
        ds.SeriesTime = ds.StudyTime

        ds.Rows, ds.Columns = nii_array.shape[1], nii_array.shape[2]
        ds.NumberOfFrames = nii_array.shape[0]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.ImageOrientationPatient = [str(x) for x in direction[:6]]
        ds.ImagePositionPatient = [str(x) for x in origin]
        # ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

        ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')

        # Ensure GridFrameOffsetVector and DoseGridScaling are correctly calculated
        ds.GridFrameOffsetVector = [float(spacing[2]) * i for i in range(nii_array.shape[0])]
        print(ds.GridFrameOffsetVector)
        ds.DoseGridScaling = np.max(nii_array) / np.iinfo(np.uint16).max

        ds.FrameIncrementPointer = pydicom.tag.Tag(0x3004, 0x000C)

        # Convert the NIfTI array to a byte string and set the PixelData element
        scaled_pixel_data = (nii_array / ds.DoseGridScaling).astype(np.uint16)
        ds.PixelData = scaled_pixel_data.tobytes()
        
        
        ds.FrameOfReferenceUID = self.FrameOfReferenceUID
        # # Create Referenced RT Plan Sequence
        referenced_rt_plan_sequence = Dataset()
        referenced_rt_plan_sequence.ReferencedSOPClassUID = pydicom.uid.RTPlanStorage
        if rtrp_uid is None:
            referenced_rt_plan_sequence.ReferencedSOPInstanceUID = pydicom.uid.generate_uid()#TODO: link RP
        else:
            referenced_rt_plan_sequence.ReferencedSOPInstanceUID = rtrp_uid
        ds.ReferencedRTPlanSequence = [referenced_rt_plan_sequence]
        
        # # Create Referenced RT Structure Sequence
        referenced_rt_rs_sequence = Dataset()
        referenced_rt_rs_sequence.ReferencedSOPClassUID = pydicom.uid.RTStructureSetStorage
        if rtrs_uid is None:
            referenced_rt_rs_sequence.ReferencedSOPInstanceUID = pydicom.uid.generate_uid()
        else:
            referenced_rt_rs_sequence.ReferencedSOPInstanceUID = rtrs_uid

        ds.ReferencedStructureSetSequence = [referenced_rt_rs_sequence]
        

        # Save the DICOM RT Dose file
        try:
            ds.save_as(output_path)
            print(f"Converted DICOM RT Dose saved to {output_path}")
        except Exception as e:
            print(f"Error saving DICOM file: {e}")
            
            
    def read_and_display_slice(self, dicom_file=None, axies=0):
        # Load the DICOM file
        if dicom_file is None:
            print("default: read converted CT")
            slice_index=int(len(os.listdir(self.CT_output_folder))//2)
            
            dicom_file=os.path.join(self.CT_output_folder,sorted(os.listdir(self.CT_output_folder))[slice_index])
        ds = pydicom.dcmread(dicom_file)
        
        # Print out some key metadata
        print("Patient Name:", ds.PatientName)
        print("Patient ID:", ds.PatientID)
        print("Modality:", ds.Modality)
        print("Rows:", ds.Rows)
        print("Columns:", ds.Columns)
        print("Pixel Spacing:", ds.PixelSpacing)
        print("Slice Thickness:", ds.SliceThickness)
        print("Image Position Patient:", ds.ImagePositionPatient)
        print("Image Orientation Patient:", ds.ImageOrientationPatient)
        
        # Display the image data
        plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
        plt.title(f"Slice: {ds.SliceLocation}")
        plt.show()
    def read_and_display_img(self, file_path, slice_index=None):
        # Read the DICOM file
        dicom_dataset = pydicom.dcmread(file_path)
        if slice_index is None:
            slice_index=int(dicom_dataset.pixel_array.shape[2]//2)
        # Extract the pixel data
        pixel_array = dicom_dataset.pixel_array[slice_index]
        
        # Display the slice
        plt.figure(figsize=(10, 10))
        plt.imshow(pixel_array, cmap='gray')
        plt.title(f'Slice {slice_index}')
        plt.axis('off')
        plt.show()
