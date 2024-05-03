> [!note]
> Here is how things should look like in nnDose_raw!\
Please check **[dataset_format(nnDose_dicom)](dataset_format(nnDose_dicom).md)** before reading this! 
Unless you are formatting your own dataset,then please carefully read this page.

# nnDose_raw folder structure
Datasets must be located in the `nnDose_raw` folder (which you either define when installing nnDoseNet or export/set every time you intend to run nnDoseNet commands!).
Each labeled dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name that correspond to the folder in `nnDose_dicom` (you can change it to other name, it doesn't need to be the same in nnDose_dicom) 

    nnDose_raw/
    ├── Dataset001_open_kbp
    ├── Dataset002_open_kbp
    ├── Dataset003_UAB_prostate
    ├── Dataset004_openkbp
    ├── ...
    ├── Dataset005_UAB_prostate
        ├── structure.json  # generate by previous process, not required
        ├── patientTr.csv   # generate by previous process, not required
        ├── dataset.json
        ├── imagesTr
        └── labelsTr


## imagesTr
Contains the images belonging to the training cases. nnDoseNet will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
Folder structure should look like this in imagesTr:

    imagesTr
        ├── 001_0000.nii.gz
        ├── 001_0001.nii.gz
        ├── 001_0002.nii.gz
        ├── 001_0003.nii.gz
        ├── 002_0000.nii.gz
        ├── 002_0001.nii.gz
        ├── 002_0002.nii.gz
        ├── 002_0003.nii.gz
        ├── ...

First 3 digit represent the patient number, that is anonymized while formatting it from nnDose_dicom folder. Please check the section **patinet__.csv** in  **[dataset_format(nnDose_dicom)](documentation/dataset_format(nnDose_dicom).md)**. \
And each image file in imagesTr represent certain modality .

- **_0000** `CT` : a full image(you can use MR or other instead, just rememeber to change it in dataset.json).
- **_0001** `Targets` : labeled targets.
- **_0002** `OARs` : labeled OARs.
- **_0003** `Body` : binary mask of body contour or dose range.


## labelsTr
contains the images with the ground truth dose maps for the training cases. 
Folder structure should look like this in labelsTr:

    labelsTr
        ├── 001.nii.gz
        ├── 002.nii.gz
        ├── ...

## dataset.json
Inside dataset.json should look like this. "CT" in "channel_name" should be changed if using other modality.(*please check [normalization](channel_modality_normalization) if you have question about modality operation )\
*Please don't change "labels" and "mask_channel".

    {
        "labels": {"background": "0", "dose": ""},
        "channel_names": {
            "0": "CT",
            "1": "Target",
            "2": "OAR",
            "3": "BODY"
        },
        "numTraining": 47,
        "file_ending": ".nii.gz",
        "mask_channel": [ 1, 2, 3 ]
    }

We tend to keep the way [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) does so it might be a little strange that we keep labels, since dose prediction would only have one background and dose in label. 

# Related command
 - `nnDoseNet_plan_and_preprocess`
 - `nnDoseNet_plan_experiment`
 - `nnDoseNet_preprocess`