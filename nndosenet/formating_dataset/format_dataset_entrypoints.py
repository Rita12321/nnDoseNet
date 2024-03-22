from nndosenet.configuration import default_num_processes
from nndosenet.formating_dataset.format_dataset_api import format_dataset_from_dicom, format_dataset_from_nifti,organize_dicom
from nndosenet.paths import nnUNet_raw,nnUNet_dicom
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
     raw_dicom_folder = join(nnUNet_dicom,args.name)
     
     # if args.trts =='Both':
          
     # else:
     organize_dicom(join(raw_dicom_folder,args.trts))
     format_dataset_from_dicom(join(raw_dicom_folder,args.trts))
          
          
     

def label_nifti_and_format_dataset_entry():#nnUNetv2_plan_experiment
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-d', nargs='+', type=int,
                         help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                              "planning and preprocessing for these datasets. Can of course also be just one dataset")

     args, unrecognized_args = parser.parse_known_args()
     format_dataset_from_nifti(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                         args.overwrite_plans_name,args.dose)

def format_dataset_entry():#nnUNetv2_plan_and_preprocess
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-d', nargs='+', type=int,
                         help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                              "planning and preprocessing for these datasets. Can of course also be just one dataset")
     args, unrecognized_args = parser.parse_known_args()
     # fingerprint extraction
     organize_dicom()
     print("Fingerprint extraction...")
     format_dataset_from_dicom(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.dose, args.clean, args.verbose)

     # experiment planning
     print('Experiment planning...')        
     print(args.dose)
     format_dataset_from_nifti(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name, args.dose)

     # manage default np     
     preprocess(args.d, args.overwrite_plans_name, args.c, np, args.verbose)

if __name__ == '__main__':
    organize_and_convert_dicom_entry()
