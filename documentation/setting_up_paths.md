# Setting up Paths

nnDoseNet relies on environment variables to know where raw data, preprocessed data and trained model weights are stored. 
To use the full functionality of nnU-Net, the following three environment variables must be set:

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
                        ├──Nifti                    
                            ├──ct.nii.gz/img.nii.gz
                            ├──dose.nii.gz
                            ├──PTV1.nii.gz
                            ├──PTV2.nii.gz
                            ├──...
                            ├──OAR1.nii.gz
                            ├──OAR2.nii.gz
                            ├──...
                            ├──body.nii.gz
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
throughput (such as a nvme SSD (PCIe gen 3 is sufficient)).

4) `nnDose_results`: This specifies where nnDoseNet will save the model weights. If pretrained models are downloaded, this is where it will save them.

5) `nnDose_prediction`: This is where the inference files located

### How to set environment variables
See [here](set_environment_variables.md).