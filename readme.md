
# What is nnDoseNet?
nnDoseNet is a U-Net based deep learning framework for **dose prediction** in radiotherapy(RT) planing.\
User can easily train a model given dicom or nifti data, and the nnDose would automatically convert, label, analysis, preprocess and make training plan based on the input dataset.\
It is built on top of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), a deep learning framework for segmenation purpose.

# How flexible?
- **Various dataset**: 2D/3D images, different modality, different site, DICM/Nifti and challenge/clinical dataset.

- **Cross institution/prescription**: Able to train with different RT planning guideline and prescription of Targets and OARs following user's instruction in structure.json.

- **Training parameter**: Train in different loss, epoch number and UNet structure by changing single argument. Patch size, batch size, depth ... can also easily change in plans.json.

# What's for RT planning?
**Convertion pipeline**: 
extract stucture set, ct and dose map from dicom format into nifti and label  and combine the structure for following process in nnDoseNet.

**Clical goal**: 
Label the Targets and OARs with corresponding prescription and clical goal. 

**DVH-based loss**:
With given CT image(or other modality), Targets, OARs, and body contour(optinal), nnDoseNet provide DVH-based and masked-MAE/MSE loss for clical acceptable dose plan.

**RT metrices**: such as D_0.1cc, D_95, D_99, D_1, mean, min and max. For futher analysis, nnDoseNet also come with  gamma analysis function (dicom dataset required).

# How does nnDoseNet work?
#### 1. Convertion
nnDoseNet first convert the provided dicom/nifti under `nnDose_dicom` into 4 seperate Nifti image composed dataset in `nnDose_raw`\

- **_0000** `CT` : a full image, prefer CT, other modality also work.
- **_0001** `Targets` : labeled targets with given prescription in structure.json.
- **_0002** `OARs` : labeled OARs with given prescription in structure.json.
- **_0003** `Body` : binary mask of body contour or dose range, if not provided would set 1 to full image by default for now.

[!NOTE] 
Please check **structure.json** section in **[dataset_format](documentation/dataset_format.md)**

#### 2. Prerpocess
nnDoseNet will systematically analyze the provided training cases under `nnDose_raw` and create a 'dataset fingerprint' following nnUNet pipieline, 
nnDoseNet then creates several U-Net configurations for each dataset in `nnDose_preprocessed`: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres`: a 3D U-Net operates on low resolution images (for 3D datasets with large image sizes only)

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

## What happened to the old nnU-Net?
The core of the old nnU-Net was hacked together in a short time period while participating in the Medical Segmentation 
Decathlon challenge in 2018. Consequently, code structure and quality were not the best. Many features 
were added later on and didn't quite fit into the nnU-Net design principles. Overall quite messy, really. And annoying to work with.

nnU-Net V2 is a complete overhaul. The "delete everything and start again" kind. So everything is better 
(in the author's opinion haha). While the segmentation performance [remains the same](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing), a lot of cool stuff has been added. 
It is now also much easier to use it as a development framework and to manually fine-tune its configuration to new 
datasets. A big driver for the reimplementation was also the emergence of [Helmholtz Imaging](http://helmholtz-imaging.de), 
prompting us to extend nnU-Net to more image formats and domains. Take a look [here](documentation/changelog.md) for some highlights.

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

