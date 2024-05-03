# Data format for Inference 
>[!notice]
>Please check **[dataset_format(nnDose_dicom)](dataset_format(nnDose_dicom).md)** and **[dataset_format(nnDose_raw)](dataset_format(nnDose_raw).md)** before reading this, Thank you :)

# Ts folder in nnDose_dicom
If the inference data is a split of training data in nnDose_dicom/[Dataset Name]/Ts and it's the same format of Tr(training) data. Please use one of these command for formatting dataset.(*-trts is set to Ts)

Start from  [Dicom dataset format 1 (unorganized)](dataset_format(nnDose_dicom).md):

    nnDoseNet_format_dataset -name [Dataset Name] -id [id in nnDose_raw] -trts Ts -nps [num of processor]

Start from  [Dicom dataset format 2 (organized)](dataset_format(nnDose_dicom).md):

    nnDoseNet_format_dataset -name [Dataset Name] -id [id in nnDose_raw] -trts Ts -nps [num of processor] --organized_dicom

Start from [Nifti dataset](dataset_format(nnDose_dicom).md):

    nnDoseNet_label_nifti_and_final_formatting -name [Dataset Name] -id [id in nnDose_raw] -trts Ts -nps [num of processor] 
    
After runnning the convertion code the dataset would be formatted and moved to `nnDose_prediction/[Dataset Name]/[Dataset_xxx]/GroundTruth`
Inside folder GroundTruth, There will be a patientTs.csv, folder imagesTs and folder labelsTs. The data format of these 2 folder would match the one in nnDose_raw. So no need to worry about the below section.[Jump to evaluation !](#evaluation)

    nnDosePrediction/UAB_prostate/Dataset005
    ├── patientTs.csv   # generate by previous process, not required
    ├── imagesTr
    └── labelsTr
# Outsider  folder
The data format for inference must match the one used in nnDose_raw (**specifically, the images must be in exactly 
the same format as in the imagesTr folder**). As before, the filenames must start with a unique identifier, followed by a 4-digit modality identifier. Here is an example for two different datasets:

In current version of nnDoseNet, there is always 4 modality of images:

- **_0000** `CT` : a full image(you can use MR or other instead, just rememeber to change it in dataset.json).
- **_0001** `Targets` : labeled targets.
- **_0002** `OARs` : labeled OARs.
- **_0003** `Body` : binary mask of body contour or dose range.

The data format of folder:

    input_folder
        ├── 001_0000.nii.gz
        ├── 001_0001.nii.gz
        ├── 001_0002.nii.gz
        ├── 001_0003.nii.gz
        ├── 002_0000.nii.gz
        ├── 002_0001.nii.gz
        ├── 002_0002.nii.gz
        ├── 002_0003.nii.gz
        ├── ...
Please use this code for prediction
            

Remember that the file format used for inference must be the same as was used for training.

# Evaluation

# Convert back to dicom

# Gamma analysis


   