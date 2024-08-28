
# What is nnDoseNet?
nnDoseNet is a U-Net based deep learning framework for **dose prediction** in radiotherapy(RT) planing.\
User can easily train a model given dicom or nifti data, and the nnDose would automatically convert, label, analysis, preprocess and make training plan based on the input dataset.\
It is built on top of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), a deep learning framework for segmentation purpose.

# How flexible can nnDoseNet be?
- **Various dataset**: 2D/3D images, different modality, different site, DICOM/Nifti and challenge/clinical dataset.

- **Cross institution/prescription**: Able to train with different RT planning guideline and prescription of Targets and OARs following user's instruction in structure.json.

- **Training parameter**: Train in different loss, epoch number and UNet structure by changing single argument. Patch size, batch size, depth ... can also easily change in plans.json.

# Customized for RT planning?
- **Conversion pipeline**: 
extract structure set, ct and dose map from dicom format into nifti and label  and combine the structure for following process in nnDoseNet.

- **Clinical goal**: 
Label the Targets and OARs with corresponding prescription and clinical goal. 

- **DVH-based loss**:
With given CT image(or other modality), Targets, OARs, and body contour(optional), nnDoseNet provide DVH-based and masked-MAE/MSE loss for clinical acceptable dose plan.

- **RT metrics**: such as D_0.1cc, D_95, D_99, D_1, mean, min and max. For further analysis, nnDoseNet also come with  gamma analysis function (dicom dataset required).

# How does nnDoseNet work?
### 1. Conversion & Label
nnDoseNet first convert the provided dicom/nifti under `nnDose_dicom` into 4 separate Nifti image composed dataset in `nnDose_raw`
- **_0000** `CT` : a full image, prefer CT, other modality also work.
- **_0001** `Targets` : labeled targets with given prescription in structure.json.
- **_0002** `OARs` : labeled OARs with given clinical goal in structure.json.
- **_0003** `Body` : binary mask of body contour or dose range, if not provided would set 1 to full image by default for now.


> [!note]
> Please check **structure.json** section in 
**[dataset_format(nnDose_dicom)](documentation/dataset_format(nnDose_dicom).md)** and 
**[dataset_format(nnDose_raw)](documentation/dataset_format(nnDose_raw).md)** 


### 2. Analysis & Preprocess

nnDoseNet will systematically analyze the provided training cases under `nnDose_raw` and create a 'dataset fingerprint' following nnUNet pipeline, 
nnDoseNet then creates several U-Net configurations for each dataset in `nnDose_preprocessed`: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres`: a 3D U-Net operates on low resolution images (for 3D datasets with large image sizes only)

nnDoseNet configures its prediction pipelines following nnU-Net's three-step recipe:
- **Fixed parameters** are not adapted. During some of robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, optimizer, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.


> Please check **plan.json** section in **[explanation_plans_json_files.md](documentation/explanation_plans_json_files.md)**\
> All of the above configuration can easily change by editing plan.json

### 3. Training & Validation

In default nnDoseNet use 5 cross-validation and training the model following the plan.json.Also, You can change the loss mode and total epoch num by passing argument in training command. 
In training, nnDoseNet would find the correspond configured data in `nnDose_preprocessed` as input and the trained model can be find in `nnDose_results` 

> [!note]
> Please check **Model training section at [here](documentation/how_to_use_nndosenet.md)**

### 4. Evaluation & Prediction

If all folds finish training and find best configuration, there will be a summary.json auto generate under `nnDose_results` folder. \
Then we can use prediction command and get prediction of Ts(testing files), then it would give a summary.json of Ts files. In summary.json nnDoseNet evaluate
- `MSE` of full image
- `mean`, `min`, `max` of each ROI in ground truth and prediction
- `D_0.1cc`:  The minimum dose received by the hottest sub volume of 0.1cc of each ROI in ground truth and prediction
- `D_99`, `D_95`, `D_1`: The minimum dose received by the hottest 99%, 95% and 1%  of  volume of each ROI in ground truth and prediction
 - `homogeneity`: The homogeneity of the dose in contour.
 - `conformity`: The conformity of the contour.
- `# > target prescription of D_99, D_95`: The number of cases in this run of analysis that has D_99 greater than prescription, also D_95.

All above metrics are calculated case-by-case and also calculate average of metrics among all cases.

In addition, if ground truth dicom files are provided, nnDoseNet provide the function of 
- generate difference map,
- converting prediction(Nifti) into dicom format 
- gamma analysis on those cases 

> [!note]
> Please check **[evaluation_methods.md](documentation/evaluation_methods.md)**

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)
- [all the command you need](documentation/how_to_use_nnunet.md)

Additional information:
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)



