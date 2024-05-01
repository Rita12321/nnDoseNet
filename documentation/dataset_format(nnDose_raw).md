# Dataset format
Here is how things should look like in nnDose_diom! 
# Dicom dataset format 1 (unorganized) in nnDose_dicom
If the Datatset is in Dicom format 1, please make sure 
 - structure.json is in Dataset's folder
 - data is divided to Tr for training and Ts for testing .
 - CT or full image dcms start with CT.
 - structure dcms start with RS.
 - dose map start with RD, and should on;y has ONE RD file, otherwise would only use first one. 

        nnDose_dicom
        ├──openkbp
        ├──UAB_headNneck
        ├──...
        ├──UAB_prostate       
            ├──structure.json
            ├──Ts
            ├──Tr
                ├──patient 1
                    ├──CT.12345678.91011112.dcm
                    ├──CT.12345678.91011113.dcm
                    ├──CT.12345678.91011114.dcm
                    ...
                    ├──RS.12345678.91011117.dcm
                    ├──RS.12345678.91011118.dcm
                    ...
                    └──RD.12345678.91011119.dcm
                ├──patient 2
                    ├──CT.22345678.91011112.dcm
                    ├──CT.22345678.91011113.dcm
                    ├──CT.22345678.91011114.dcm
                    ...
                    ├──RS.22345678.91011117.dcm
                    ├──RS.22345678.91011118.dcm
                    ...
                    └──RD.22345678.91011119.dcm
                ├──patient 3 
            ...
Please use this code to format dataset

        nnDoseNet_format_dataset -name [Dataset Name] -id [id in nnDose_raw] -trts [Tr or Ts] -nps [num of processor]

# Dicom dataset format 2 (organized) in nnDose_dicom
If the Datatset is in Dicom format 2, please make sure 
 - structure.json is in Dataset's folder
 - data is divided to Tr for training and Ts for testing .

        nnDose_dicom
        ├──openkbp
        ├──UAB_prostate
        ├──UAB_headNneck
        ├──...
            ├──structure.json
            ├──Ts
            ├──Tr
                ├──patient 1
                    ├──Nifti
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
                ├──patient 2
                ├──patient 3 
                ...
Please use this code to format dataset

        nnDoseNet_format_dataset -name [Dataset Name] -id [id in nnDose_raw] -trts [Tr or Ts] -nps [num of processor] --organized_dicom

# Nifti dataset in nnDose_dicom
If the Datatset is in Dicom format 2  or changing structure.json for different label, please make sure 
 - structure.json is in Dataset's folder
 - data is divided to Tr for training and Ts for testing .
 - The PTV, OAR and Body need to be the SAME name in structure.json

        nnDose_dicom
        ├──openkbp
        ├──UAB_prostate
        ├──UAB_headNneck
        ├──...
            ├──structure.json
            ├──Ts
            ├──Tr
                ├──patient 1
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
                ...
Please use this code to label and final formating the dataset

        nnDoseNet_label_nifti_and_final_formatting -name [Dataset Name] -id [id in nnDose_raw] -trts [Tr or Ts] -nps [num of processor] 
        
# structure.json

nnDoseNet use the structure.json for ROI extractiong and labeling.
There are 2 major part of structure.json, `structure_name` and `structure_label`.


**`structure_name`** is used to extract structure from dicom to nifti. The key is **ROI's final name** and value is a **list of potential name** of the ROI in dicom.\
*If the dataset is in nifti format, you don't need stucture_name.

    "structure_name":{
        "Bladder": ["Bladder"], 
        "Rectum": ["Rectum"], 
        "Femur_Head_L": ["Femur_Head_L"],
        "Femur_Head_R": ["Femur_Head_R"],
        "Bowel": ["Bowel_Large","Bowel_Small","Bag_Bowel","Bowel"],
        "High_PTV": ["PTV","PTV_High","PTV_High^p","PTV_High^p+sv","PTV_Pros_7000","PTV^p+sv","PTV1","PTVp_High"],
        "Body":["BODY","body","Body","External","external","EXTERNAL"]}    


> nnDoseNet search the dicom following the order of potenial name list and once it's found, it won't keep searching.

**`structure_label`** is used to label the structure nifti and combine it . The key has to be class `Target`, `OAR`, and `Body` which correspond to _0001, _0002, _0003 in Dataset_xxx under nnDoseNet_raw. In each class, the key is the prescription or clical goal of the correspond list of ROI.\
*Please make sure the ROI name is the same as ROI's final name in structure_name.

    "structure_label":{
        "Target": {"70": ["High_PTV"]}, 
        "OAR": {"31": ["Bladder"],"35":["Rectum"],"50":["Femur_Head_L","Femur_Head_R"],"54":["Bowel"]},
        "Body":{"1":["Body"]}}   
>Strongly suggest to put right and left contour in same value.The data augmentaion would mirror the image, so it won't tell the right and left. 

Final look should be like this, if you have diocm dataset:

    {"structure_name":{
    "Bladder": ["Bladder"], 
    "Rectum": ["Rectum"], 
    "Femur_Head_L": ["Femur_Head_L"],
    "Femur_Head_R": ["Femur_Head_R"],
    "Bowel": ["Bowel_Large","Bowel_Small","Bag_Bowel","Bowel"],
    "High_PTV": ["PTV","PTV_High","PTV_High^p","PTV_High^p+sv","PTV_Pros_7000","PTV^p+sv","PTV1","PTVp_High"],
    "Body":["BODY","body","Body","External","external","EXTERNAL"]},

    "structure_label":{
        "Target": {"70": ["High_PTV"]}, 
        "OAR": {"31": ["Bladder"],"35":["Rectum"],"50":["Femur_Head_L","Femur_Head_R"],"54":["Bowel"]},
        "Body":{"1":["Body"]}}
    }

# 
**Images** is plural because nnU-Net supports arbitrarily many input channels. In order to be as flexible as possible, 
nnU-net requires each input channel to be stored in a separate image (with the sole exception being RGB natural 
images). So these images could for example be a T1 and a T2 MRI (or whatever else you want). The different input 
channels MUST have the same geometry (same shape, spacing (if applicable) etc.) and
mus[text](dataset_format.md)t be co-registered (if applicable). Input channels are identified by nnU-Net by their FILE_ENDING: a four-digit integer at the end 
of the filename. Image files must therefore follow the following naming convention: {CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}. 
Hereby, XXXX is the 4-digit modality/channel identifier (should be unique for each modality/chanel, e.g., “0000” for T1, “0001” for 
T2 MRI, …) and FILE_ENDING is the file extension used by your image format (.png, .nii.gz, ...). See below for concrete examples.
The dataset.json file connects channel names with the channel identifiers in the 'channel_names' key (see below for details).

Side note: Typically, each channel/modality needs to be stored in a separate file and is accessed with the XXXX channel identifier. 
Exception are natural images (RGB; .png) where the three color channels can all be stored in one file (see the [road segmentation](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py) dataset as an example). 

**Segmentations** must share the same geometry with their corresponding images (same shape etc.). Segmentations are 
integer maps with each value representing a semantic class. The background must be 0. If there is no background, then 
do not use the label 0 for something else! Integer values of your semantic classes must be consecutive (0, 1, 2, 3, 
...). Of course, not all labels have to be present in each training case. Segmentations are saved as {CASE_IDENTIFER}.{FILE_ENDING} .

Within a training case, all image geometries (input channels, corresponding segmentation) must match. Between training 
cases, they can of course differ. nnU-Net takes care of that.

Important: The input channels must be consistent! Concretely, **all images need the same input channels in the same 
order and all input channels have to be present every time**. This is also true for inference!


## Supported file formats
nnU-Net expects the same file format for images and segmentations! These will also be used for inference. For now, it 
is thus not possible to train .png and then run inference on .jpg.

One big change in nnU-Net V2 is the support of multiple input file types. Gone are the days of converting everything to .nii.gz!
This is implemented by abstracting the input and output of images + segmentations through `BaseReaderWriter`. nnU-Net 
comes with a broad collection of Readers+Writers and you can even add your own to support your data format! 
See [here](../nnunetv2/imageio/readme.md).

As a nice bonus, nnU-Net now also natively supports 2D input images and you no longer have to mess around with 
conversions to pseudo 3D niftis. Yuck. That was disgusting.

Note that internally (for storing and accessing preprocessed images) nnU-Net will use its own file format, irrespective 
of what the raw data was provided in! This is for performance reasons.


By default, the following file formats are supported:
- NaturalImage2DIO: .png, .bmp, .tif
- NibabelIO: .nii.gz, .nrrd, .mha
- NibabelIOWithReorient: .nii.gz, .nrrd, .mha. This reader will reorient images to RAS!
- SimpleITKIO: .nii.gz, .nrrd, .mha
- Tiff3DIO: .tif, .tiff. 3D tif images! Since TIF does not have a standardized way of storing spacing information, 
nnU-Net expects each TIF file to be accompanied by an identically named .json file that contains three numbers 
(no units, no comma. Just separated by whitespace), one for each dimension.


The file extension lists are not exhaustive and depend on what the backend supports. For example, nibabel and SimpleITK 
support more than the three given here. The file endings given here are just the ones we tested!

IMPORTANT: nnU-Net can only be used with file formats that use lossless (or no) compression! Because the file 
format is defined for an entire dataset (and not separately for images and segmentations, this could be a todo for 
the future), we must ensure that there are no compression artifacts that destroy the segmentation maps. So no .jpg and 
the likes! 

## Dataset folder structure
Datasets must be located in the `nnUNet_raw` folder (which you either define when installing nnU-Net or export/set every 
time you intend to run nnU-Net commands!).
Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit 
integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and 
the dataset id is 5. Datasets are stored in the `nnUNet_raw` folder like this:

    nnUNet_raw/
    ├── Dataset001_BrainTumour
    ├── Dataset002_Heart
    ├── Dataset003_Liver
    ├── Dataset004_Hippocampus
    ├── Dataset005_Prostate
    ├── ...

Within each dataset folder, the following structure is expected:

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs  # optional
    └── labelsTr


When adding your custom dataset, take a look at the [dataset_conversion](../nnunetv2/dataset_conversion) folder and 
pick an id that is not already taken. IDs 001-010 are for the Medical Segmentation Decathlon.

- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these images. Remnant of the Medical Segmentation Decathlon folder structure.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.

The scheme introduced [above](#what-do-training-cases-look-like) results in the following folder structure. Given 
is an example for the first Dataset of the MSD: BrainTumour. This dataset hat four input channels: FLAIR (0000), 
T1w (0001), T1gd (0002) and T2w (0003). Note that the imagesTs folder is optional and does not have to be present.

    nnUNet_raw/Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    │   ├── BRATS_001_0000.nii.gz
    │   ├── BRATS_001_0001.nii.gz
    │   ├── BRATS_001_0002.nii.gz
    │   ├── BRATS_001_0003.nii.gz
    │   ├── BRATS_002_0000.nii.gz
    │   ├── BRATS_002_0001.nii.gz
    │   ├── BRATS_002_0002.nii.gz
    │   ├── BRATS_002_0003.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── BRATS_485_0000.nii.gz
    │   ├── BRATS_485_0001.nii.gz
    │   ├── BRATS_485_0002.nii.gz
    │   ├── BRATS_485_0003.nii.gz
    │   ├── BRATS_486_0000.nii.gz
    │   ├── BRATS_486_0001.nii.gz
    │   ├── BRATS_486_0002.nii.gz
    │   ├── BRATS_486_0003.nii.gz
    │   ├── ...
    └── labelsTr
        ├── BRATS_001.nii.gz
        ├── BRATS_002.nii.gz
        ├── ...

Here is another example of the second dataset of the MSD, which has only one input channel:

    nnUNet_raw/Dataset002_Heart/
    ├── dataset.json
    ├── imagesTr
    │   ├── la_003_0000.nii.gz
    │   ├── la_004_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── la_001_0000.nii.gz
    │   ├── la_002_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── la_003.nii.gz
        ├── la_004.nii.gz
        ├── ...

Remember: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also 
make sure that all your data is co-registered!

See also [dataset format inference](dataset_format_inference.md)!!

## dataset.json
The dataset.json contains metadata that nnU-Net needs for training. We have greatly reduced the number of required 
fields since version 1!

Here is what the dataset.json should look like at the example of the Dataset005_Prostate from the MSD:

    { 
     "channel_names": {  # formerly modalities
       "0": "T2", 
       "1": "ADC"
     }, 
     "labels": {  # THIS IS DIFFERENT NOW!
       "background": 0,
       "PZ": 1,
       "TZ": 2
     }, 
     "numTraining": 32, 
     "file_ending": ".nii.gz"
     "overwrite_image_reader_writer": "SimpleITKIO"  # optional! If not provided nnU-Net will automatically determine the ReaderWriter
     }

The channel_names determine the normalization used by nnU-Net. If a channel is marked as 'CT', then a global 
normalization based on the intensities in the foreground pixels will be used. If it is something else, per-channel 
z-scoring will be used. Refer to the methods section in [our paper](https://www.nature.com/articles/s41592-020-01008-z) 
for more details. nnU-Net v2 introduces a few more normalization schemes to 
choose from and allows you to define your own, see [here](explanation_normalization.md) for more information. 

Important changes relative to nnU-Net v1:
- "modality" is now called "channel_names" to remove strong bias to medical images
- labels are structured differently (name -> int instead of int -> name). This was needed to support [region-based training](region_based_training.md)
- "file_ending" is added to support different input file types
- "overwrite_image_reader_writer" optional! Can be used to specify a certain (custom) ReaderWriter class that should 
be used with this dataset. If not provided, nnU-Net will automatically determine the ReaderWriter
- "regions_class_order" only used in [region-based training](region_based_training.md)

There is a utility with which you can generate the dataset.json automatically. You can find it 
[here](../nnunetv2/dataset_conversion/generate_dataset_json.py). 
See our examples in [dataset_conversion](../nnunetv2/dataset_conversion) for how to use it. And read its documentation!

## How to use nnU-Net v1 Tasks
If you are migrating from the old nnU-Net, convert your existing datasets with `nnUNetv2_convert_old_nnUNet_dataset`!

Example for migrating a nnU-Net v1 Task:
```bash
nnUNetv2_convert_old_nnUNet_dataset /media/isensee/raw_data/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC Dataset027_ACDC 
```
Use `nnUNetv2_convert_old_nnUNet_dataset -h` for detailed usage instructions.


## How to use decathlon datasets
See [convert_msd_dataset.md](convert_msd_dataset.md)

## How to use 2D data with nnU-Net
2D is now natively supported (yay!). See [here](#supported-file-formats) as well as the example dataset in this 
[script](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py).


## How to update an existing dataset
When updating a dataset it is best practice to remove the preprocessed data in `nnUNet_preprocessed/DatasetXXX_NAME` 
to ensure a fresh start. Then replace the data in `nnUNet_raw` and rerun `nnUNetv2_plan_and_preprocess`. Optionally, 
also remove the results from old trainings.

# Example dataset conversion scripts
In the `dataset_conversion` folder (see [here](../nnunetv2/dataset_conversion)) are multiple example scripts for 
converting datasets into nnU-Net format. These scripts cannot be run as they are (you need to open them and change 
some paths) but they are excellent examples for you to learn how to convert your own datasets into nnU-Net format. 
Just pick the dataset that is closest to yours as a starting point.
The list of dataset conversion scripts is continually updated. If you find that some publicly available dataset is 
missing, feel free to open a PR to add it!
