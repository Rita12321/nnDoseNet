# Setting up Paths

nnDoseNet relies on environment variables to know where dicom data, raw data, preprocessed data, trained model weights and inference data are stored. 
To use the full functionality of nnDoseNet the following 5 environment variables must be set:

1) `nnDose_dicom`: This is where you put dicoms or unlabeled niftis(The contours are still individually exist in binary maps ). It's too much to include here please check [here](dataset_format(nnDose_dicom).md).


    Example tree structure:

            nnDose_dicom
            ├──openkbp
            ├──UAB_prostate
            ├──NAME3
            ├──...
                ├──structure.json
                ├──Ts
                ├──Tr
                    ├──patient 1

                        ├──Dicom
                            ├──CT
                                ├──CT.12345678.91011112.dcm
                                ├──CT.12345678.91011113.dcm
                                ├──CT.12345678.91011114.dcm
                            ├──RS
                                ├──RS.12345678.91011117.dcm
                                ├──RS.12345678.91011118.dcm
                            ├──RD
                                └──RD.12345678.91011119.dcm 
                            ├──RP
                            └──Extras
                                └──xx.12345678.91011119999.dcm      

                        ├──Nifti                    
                            ├──ct.nii.gz/img.nii.gz
                            ├──dose.nii.gz
                            ├──PTV1.nii.gz
                            ├──PTV2.nii.gz
                            ├──...
                            ├──OAR1.nii.gz
                            ├──OAR2.nii.gz
                            ├──...
                            └──body.nii.gz
                    ├──patient 2
                    ├──patient 3 

2) `nnDose_raw`: This is where you place the labeled raw datasets. This folder will have one subfolder for each dataset names 
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique) 
dataset name. The datasets must be in nnDoseNet format, see [here](dataset_format(nnDose_raw).md).

    Example tree structure:
    ```
    nnDose_raw/Dataset001_openkbp
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    └── labelsTr
        ├── ...
    nnDose_raw/Dataset002_NAME2
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    └── labelsTr
        ├── ...
    ```

3) `nnDose_preprocessed`: This is the folder where the preprocessed data will be saved. The data will also be read from 
this folder during training. It is important that this folder is located on a drive with low access latency and high 
throughput. 

    Example tree structure:
    ```
    nnDose_preprocessed/Dataset001_openkbp
    ├── structure.json
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── nnDosePlans.json
    ├── gt_segmentations
        ├── ...
    ├── nnDosePlans_2d
        ├── ...
    ├── ...    
    └── nnDosePlans_3d_fullres
        ├── ...

 
    ```
    Files and folders in nnDose_preprocessed:
    - structure.json: Auto-copied. Please check [here](dataset_format(nnDose_dicom).md) for more info.
    - dataset.json: Auto-copied. Please check [here](dataset_format(nnDose_raw).md) for more info.
    - dataset_fingerprint.json: Auto-generated when planning or extracting data fingerprint. This json store the dimensions(shape and spacing) of each training data.
    - `nnDosePlans.json`: Auto-generated when planning. This json store the new dimension info and training parameter of each configuration(2d, 3d, custom).Please check [here](explanation_plans_json_files.md) for more info.
    - gt_segmentations: Just a folder with ground truths.
    - nnDosePlans_XX: The folder where stores the ready-to-fed data for training. Some configuration share the same folder, please check [here](explanation_plans_json_files.md) for info about custom configuration

4) `nnDose_results`: This specifies where nnDoseNet will save the model weights. If pretrained models are downloaded, this is where it will save them.

5) `nnDose_prediction`: This is where the inference files locate and analyzed.

### How to set environment variables
See [here](set_environment_variables.md).