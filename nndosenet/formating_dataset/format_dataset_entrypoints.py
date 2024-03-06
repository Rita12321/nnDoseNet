from nndosenet.configuration import default_num_processes
from nndosenet.formating_dataset.format_dataset_api import format_dataset_from_dicom, format_dataset_from_nifti,organize_folder,move_to_raw



def organize_and_convert_dicom_entry():#nnUNetv2_extract_fingerprint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
   
    organize_and_convert_dicom(args.d, )


def label_nifti_and_format_dataset_entry():#nnUNetv2_plan_experiment
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    
    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                     args.overwrite_plans_name,args.dose)





def format_dataset_entry():#nnUNetv2_plan_and_preprocess
    import argparse
    parser = argparse.ArgumentParser()
    

    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.dose, args.clean, args.verbose)

    # experiment planning
    print('Experiment planning...')        
    print(args.dose)
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name, args.dose)

    # manage default np
    

    
        
     preprocess(args.d, args.overwrite_plans_name, args.c, np, args.verbose)


if __name__ == '__main__':
    format_dataset_entry()
