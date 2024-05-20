from nndosenet.configuration import default_num_processes
from nndosenet.formating_dataset.format_dataset_api import format_dataset_from_dicom, format_dataset_from_nifti,organize_dicom,organize_nifti
from nndosenet.paths import nnDose_raw,nnDose_dicom
from batchgenerators.utilities.file_and_folder_operations import *

                    
                    
def organize_and_convert_dicom_entry():
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('-name', type=str,
                         help="[REQUIRED] The name of data folder in nndose_dicom you want to convert. Example: openkbp. ")
     parser.add_argument('-trts', type=str,default='Tr',
                         help="Enter 'Tr' for Training, 'Ts' for Testing ")
     parser.add_argument('-nps', type=int ,default=3,
                         help="")
     parser.add_argument('-structure', type=str ,default=None,
                         help="You can specify the structure.json you want to use in stucture convertion, default would use the structure.json\
                              in nnDose_dicom/[--name] ")     
     parser.add_argument('--organized_dicom', default=False, action="store_true", required=False,
                         help="if the dicom file is already organized or put tnto correspond folder,\
                              you will get warning about not finding dcm in folder while dcms are already moved.\
                              it's very noisy :(")
   
     args, unrecognized_args = parser.parse_known_args()
     raw_dicom_folder = join(nnDose_dicom,args.name)
     if not args.organized_dicom:
          organize_dicom(join(raw_dicom_folder,args.trts))
     format_dataset_from_dicom(raw_dicom_folder,args.trts,args.nps,args.structure)
          

def label_nifti_and_final_formatting_entry():
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
     parser.add_argument('-structure', type=str ,default=None,
                         help="You can specify the structure.json you want to use in stucture label, default would use the structure.json\
                              in nnDose_dicom/[--name] ") 
     
     args, unrecognized_args = parser.parse_known_args()
     
     # print(nnDose_raw)
     # print(nnDose_dicom)
     raw_dicom_folder = join(nnDose_dicom,args.name)

     organize_nifti(join(raw_dicom_folder,args.trts))
     format_dataset_from_nifti(join(raw_dicom_folder,args.trts), args.id,dataset_name_only=args.name,tr_ts=args.trts,num_processes=args.nps,structure_label=args.structure)

def format_dataset_entry():
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
     parser.add_argument('-structure', type=str ,default=None,
                         help="You can specify the structure.json you want to use in stucture convertion and label , default would use the structure.json\
                              in nnDose_dicom/[--name] ")
     parser.add_argument('--organized_dicom', default=False, action="store_true", required=False,
                    help="if the dicom file is already organized or put tnto correspond folder,\
                         you will get warning about not finding dcm in folder while dcms are already moved.\
                         it's very noisy :(")
     
     args, unrecognized_args = parser.parse_known_args()
     raw_dicom_folder = join(nnDose_dicom,args.name)
     
     if not args.organized_dicom:
          organize_dicom(join(raw_dicom_folder,args.trts))
     format_dataset_from_dicom(raw_dicom_folder,args.trts,args.nps,args.structure)
     organize_nifti(join(raw_dicom_folder,args.trts))
     format_dataset_from_nifti(join(raw_dicom_folder,args.trts), args.id,dataset_name_only=args.name,tr_ts=args.trts,num_processes=args.nps,structure_label=args.structure )

if __name__ == '__main__':
    format_dataset_entry()
