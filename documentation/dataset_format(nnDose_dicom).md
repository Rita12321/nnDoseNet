> [!note]
> Here is how things should look like in nnDose_diom! 
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

# roi_summary
If using dicom dataset,
after running `nnDoseNet_format_dataset` or `nnDoseNet_organize_and_convert_dicom` you should see "Tr_roi_summary.csv" or "Ts_roi_summary.csv" and it's correspond json files.\
These file summarize which name in potential name list is converted into nifti and shows the remaining structure in structure dcms that is not being used. It also, shows which patient has missing contour.

# patinet__.csv
After running `nnDoseNet_format_dataset` or `nnDoseNet_label_nifti_and_final_formatting`, you should see patinetTr.csv in nnDose_raw/Datasetxxx or patinetTs.csv in nnDose_prediction/Datasetxxx/GroundTruth.\
nnDoseNet anonymize the cases name to number, and correspond id and orginal name is in the csv.

# All command related command
- `nnDoseNet_format_dataset` 
- `nnDoseNet_organize_and_convert_dicom` 
- `nnDoseNet_label_nifti_and_final_formatting`

*use -h to check how to use the command
#
> Please check **[dataset_format(nnDose_raw)](documentation/dataset_format(nnDose_raw).md)** for next step : )