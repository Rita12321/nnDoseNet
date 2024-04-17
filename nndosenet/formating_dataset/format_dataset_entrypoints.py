from nndosenet.configuration import default_num_processes
from nndosenet.formating_dataset.format_dataset_api import format_dataset_from_dicom, format_dataset_from_nifti,organize_dicom,organize_nifti
from nndosenet.paths import nnDose_raw,nnDose_dicom
from batchgenerators.utilities.file_and_folder_operations import *

                    
                    
def organize_and_convert_dicom_entry():#nnUNetv2_extract_fingerprint
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-name', type=str,
                         help="[REQUIRED] The name of folder in nndose_dicom you want to convert. Example: openkbp. ")
     parser.add_argument('-trts', type=str,default='Tr',
                         help="Enter 'Tr' for Training, 'Ts' for Testing, 'Both' for both train and test ")
     parser.add_argument('-nps', type=int ,default=3,
                         help="")
     
     args, unrecognized_args = parser.parse_known_args()
     raw_dicom_folder = join(nnDose_dicom,args.name)
     
     organize_dicom(join(raw_dicom_folder,args.trts))
     format_dataset_from_dicom(join(raw_dicom_folder,args.trts))
          

def label_nifti_and_final_formatting_entry():#nnUNetv2_plan_experiment
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-name', type=str,
                         help="[REQUIRED] The name of folder in nndose_dicom you want to convert. Example: openkbp. ")
     parser.add_argument('-id', type=str,
                         help="[REQUIRED] The id of of the dataset  you want to name. Example: 012. ")     
     parser.add_argument('-trts', type=str,default='Tr',
                         help="Enter 'Tr' for Training, 'Ts' for Testing, 'Both' for both train and test ")
     parser.add_argument('-nps', type=int ,default=3,
                         help="")
     
     args, unrecognized_args = parser.parse_known_args()
     
     # print(nnDose_raw)
     # print(nnDose_dicom)
     raw_dicom_folder = join(nnDose_dicom,args.name)

     organize_nifti(join(raw_dicom_folder,args.trts))
     format_dataset_from_nifti(join(raw_dicom_folder,args.trts), args.id,args.name,args.trts )

def format_dataset_entry():#nnUNetv2_plan_and_preprocess
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-name', type=str,
                         help="[REQUIRED] The name of folder in nndose_dicom you want to convert. Example: openkbp. ")
     parser.add_argument('-id', type=str,
                         help="[REQUIRED] The id of of the dataset  you want to name. Example: 012. ")     
     parser.add_argument('-trts', type=str,default='Tr',
                         help="Enter 'Tr' for Training, 'Ts' for Testing, 'Both' for both train and test ")
     parser.add_argument('-nps', type=int ,default=3,
                         help="")
     
     args, unrecognized_args = parser.parse_known_args()
     raw_dicom_folder = join(nnDose_dicom,args.name)
     
     organize_dicom(join(raw_dicom_folder,args.trts))
     format_dataset_from_dicom(join(raw_dicom_folder,args.trts))
     organize_nifti(join(raw_dicom_folder,args.trts))
     format_dataset_from_nifti(join(raw_dicom_folder,args.trts), args.id,args.name,args.trts )

if __name__ == '__main__':
    #organize_and_convert_dicom_entry()
    label_nifti_and_final_formatting_entry()
