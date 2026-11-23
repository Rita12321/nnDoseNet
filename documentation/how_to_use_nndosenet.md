# Current CLI names

This historical guide may use obsolete dotted command names. The installed legacy command names are listed in [CLI_REFERENCE_LEGACY.md](CLI_REFERENCE_LEGACY.md); the maintained run order and templates are in [HANDOFF_NNDOSEV2.md](HANDOFF_NNDOSEV2.md).

## How to run nnDoseNet on a new dataset
Given some dataset, nnDoseNet fully automatically configures an entire RT dose prediction pipeline that matches its properties.
nnDoseNet covers the entire pipeline: 
1) dataset conversion from DICOM to NIfTI, extracting user-assigned structure and label the structure with user-defined clinical goals
2) preprocessing to model configuration, 
3) model training, 
4) ensembling and
5) calculate RT evaluation metrics on inference data.
After running, the trained model(s) can be applied to the test cases for inference.

### Dataset Format
nnDoseNet expects datasets in a structured format. Please read
[dataset_format(nnDose_dicom)](dataset_format(nnDose_dicom).md) and [dataset_format(nnDose_raw)](dataset_format(nnDose_raw).md) for information on how to set up datasets with DICOM or NIfTI format.

If converting dataset from DICOM please use this command:
```bash
nnDoseNet_1_format_dataset -name [Dataset Name] -id [id in nnDose_raw] -trts Tr -nps [num of processor] --organized_dicom
```
If converting dataset from NIfTI please use this command:
```bash
nnDoseNet_1.2_label_nifti_and_final_formatting -name [Dataset Name] -id [id in nnDose_raw] -trts Tr -nps [num of processor] 
```
### Experiment planning and preprocessing
Given a new dataset, nnDoseNet will extract a dataset fingerprint (a set of dataset-specific properties such as
image sizes, voxel spacings, intensity information etc) just like what nnU-Net does. This information is used to design three default configurations(2d, 3d_fullres and 3d_lowres in some dataset ). 
Each of these pipelines operates on its own preprocessed version of the dataset. For additional configuration, see [the plan-file examples](explanation_plans_json_files.md).

The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
nnDoseNet_2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Where `DATASET_ID` is the dataset id. We recommend `--verify_dataset_integrity` whenever it's the first time 
you run this command. This will check for some of the most common error sources!

You can also process several datasets at once by giving `-d 1 2 3 [...]`. If you already know what U-Net configuration 
you need you can also specify that with `-c 3d_fullres` (make sure to adapt -np in this case!). For more information 
about all the options available to you please run `nnDoseNet_2_plan_and_preprocess -h`.

nnDoseNet_2_plan_and_preprocess will create a new subfolder in your nnUNet_preprocessed folder named after the dataset. 
Once the command is completed there will be a dataset_fingerprint.json file as well as a nnUNetPlans.json file for you to look at 
(in case you are interested!). There will also be subfolders containing the preprocessed data for your UNet configurations.

[Optional]
If you prefer to keep things separate, you can also use `nnDoseNet_2.1_extract_fingerprint`, `nnDoseNet_2.2_plan_experiment` 
and `nnDoseNet_2.3_preprocess` in ths order, just like nnU-Net. 

### Model training
#### Overview
You pick which configurations (2d, 3d_fullres, 3d_lowres) should be trained! If you have no idea 
what performs best on your data, just run all of them and let nnDoesNet identify the best one. It's up to you!

nnDoseNet trains all configurations in a 5-fold cross-validation over the training cases. This is 1) needed so that 
nnDoseNet can estimate the performance of each configuration and tell you which one should be used for your 
RT dose prediction problem and 2) a natural way of obtaining a good model ensemble (average the output of these 5 models 
for prediction) to boost performance.

You can influence the splits nnDoseNet uses for 5-fold cross-validation (see [here](manual_data_splits.md)). If you 
prefer to train a single model on all training cases, this is also possible (see below).


Training models is done with the `nnDoseNet_3_train` command. The general structure of the command is:
```bash
nnDoseNet_3_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres). DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of 
the 5-fold-cross-validation is trained.

nnDoseNet stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `--c` to the
training command.

IMPORTANT: If you plan to use `nnDoseNet_find_best_configuration` (see below) add the `--npz` flag. This makes 
nnU-Net save the outputs in npz during the final validation. They are needed for that.
If you ran initially without the `--npz` flag but now require npzs, simply rerun the validation with:
```bash
nnDoseNet_3_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```

You can specify the device nnU-net should use by using `-device DEVICE`. DEVICE can only be cpu, cuda or mps. If 
you have multiple GPUs, please select the gpu id using `CUDA_VISIBLE_DEVICES=X nnDoseNet_train [...]` (requires device to be cuda).

See `nnDoseNet_train -h` for additional options.

### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnDoseNet_3_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```

### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnDoseNet_3_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
```

### 3D low resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnDoseNet_3_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]
```


The trained models will be written to the nnDoseNet_results folder. Each training obtains an automatically generated
output folder name:

nnDoseNet_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD

For Dataset002_Heart (from the MSD), for example, this looks like this:

    nnDose_results/
    ├── Dataset002_openkbbp
        │── nnUNetTrainer__nnDosePlans__2d
        │    ├── fold_0
        │    ├── fold_1
        │    ├── fold_2
        │    ├── fold_3
        │    ├── fold_4
        │    ├── dataset.json
        │    ├── structure.json        
        │    ├── dataset_fingerprint.json
        │    └── plans.json
        └── nnUNetTrainer__nnDosePlans__3d_fullres
             ├── fold_0
             ├── fold_1
             ├── fold_2
             ├── fold_3
             ├── fold_4
             ├── dataset.json
        │    ├── structure.json 
             ├── dataset_fingerprint.json
             └── plans.json

In each model training output folder (each of the fold_x folder), the following files will be created:
- debug.json: Contains a summary of blueprint and inferred parameters used for training this model as well as a 
bunch of additional stuff. Not easy to read, but very useful for debugging ;-)
- checkpoint_best.pth: checkpoint files of the best model identified during training. Not used right now unless you 
explicitly tell nnU-Net to use it.
- checkpoint_final.pth: checkpoint file of the final model (after training has ended). This is what is used for both 
validation and inference.
- network_architecture.pdf (only if hiddenlayer is installed!): a pdf document with a figure of the network architecture in it.
- progress.png: Shows losses, pseudo dice, learning rate and epoch times of the course of the training. At the top is 
a plot of the training (blue) and validation (red) loss during training.
A proper validation takes way too long to be done each epoch. It is run at the end of the training.
- validation_raw: in this folder are the predicted validation cases after the training has finished. The summary.json file in here
  contains the validation metrics (a mean over all cases is provided at the start of the file). If `--npz` was set then 
the compressed npz format outputs (saved as .npz files) are in here as well. 

During training it is often useful to watch the progress. We therefore recommend that you have a look at the generated
progress.png when running the first training. It will be updated after each epoch.

### Using multiple GPUs for training

If multiple GPUs are at your disposal, the best way of using them is to train multiple nnU-Net trainings at once, one 
on each GPU. This is because data parallelism never scales perfectly linearly, especially not with small networks such 
as the ones used by nnU-Net.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 nnDoseNet_3_train DATASET_NAME_OR_ID 2d 0 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnDoseNet_3_train DATASET_NAME_OR_ID 2d 1 [--npz] & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnDoseNet_3_train DATASET_NAME_OR_ID 2d 2 [--npz] & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnDoseNet_3_train DATASET_NAME_OR_ID 2d 3 [--npz] & # train on GPU 3
CUDA_VISIBLE_DEVICES=4 nnDoseNet_3_train DATASET_NAME_OR_ID 2d 4 [--npz] & # train on GPU 4
# This is the only place I can dream about having 5 GPU and rich problem.
...
wait
```

**Important: The first time a training is run nnU-Net will extract the preprocessed data into uncompressed numpy 
arrays for speed reasons! This operation must be completed before starting more than one training of the same 
configuration! Wait with starting subsequent folds until the first training is using the GPU! Depending on the 
dataset size and your System this should oly take a couple of minutes at most.**

If you insist on running DDP multi-GPU training, we got you covered:

`nnDoseNet_3_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X`

Again, note that this will be slower than running separate training on separate GPUs. DDP only makes sense if you have 
manually interfered with the nnU-Net configuration and are training larger models with larger patch and/or batch sizes!

Important when using `-num_gpus`:
1) If you train using, say, 2 GPUs but have more GPUs in the system you need to specify which GPUs should be used via 
CUDA_VISIBLE_DEVICES=0,1 (or whatever your ids are).
2) You cannot specify more GPUs than you have samples in your minibatches. If the batch size is 2, 2 GPUs is the maximum!
3) Make sure your batch size is divisible by the numbers of GPUs you use or you will not make good use of your hardware.

In contrast to the old nnU-Net, DDP is now completely hassle free. Enjoy!

### Automatically determine the best configuration
Once the desired configurations were trained (full cross-validation) you can tell nnU-Net to automatically identify 
the best combination for you:

```bash
nnDoseNet_3_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

`CONFIGURATIONS` hereby is the list of configurations you would like to explore. Per default, ensembling is enabled 
meaning that nnDoseNet will generate all possible combinations of ensembles (2 configurations per ensemble). You can disable ensembling by setting the `--disable_ensembling` flag.

See `nnDoseNet_3_find_best_configuration -h` for more options.



Once completed, the command will print to your console exactly what commands you need to run to make predictions. It 
will also create two files in the `nnnnDoseNet_results/DATASET_NAME` folder for you to inspect: 
- `inference_instructions.txt` again contains the exact commands you need to use for predictions
- `inference_information.json` can be inspected to see the performance of all configurations and ensembles, as well 
as the effect of the postprocessing plus some debug information. 

### Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on 
and must adhere to the nnU-Net naming scheme for image files ([dataset_format(nnDose_dicom)](dataset_format(nnDose_dicom).md), [dataset_format(nnDose_raw)](dataset_format(nnDose_raw).md)and 
[inference data format](dataset_format_inference.md)!)

`nnDoseNet_3_find_best_configuration` (see above) will print a string to the terminal with the inference commands you need to use.
The easiest way to run inference is to simply use these commands.

If you wish to manually specify the configuration(s) used for inference, use the following commands:

#### Run prediction
For each of the desired configurations, run:
```
nnDoseNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

Only specify `--save_probabilities` if you intend to use ensembling. `--save_probabilities` will make the command save the predicted
probabilities alongside of the predicted segmentation masks requiring a lot of disk space.

Please select a separate `OUTPUT_FOLDER` for each configuration!

Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very 
strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference. 

If you wish to make predictions with a single model, train the `all` fold and specify it in `nnDoseNet_4_predict`
with `-f all`

#### Ensembling multiple configurations
If you wish to ensemble multiple predictions (typically form different configurations), you can do so with the following command:
```bash
nnDoseNet_4_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were
generated by `nnDoseNet_4_predict`. Again, `nnDoseNet_4_ensemble -h` will tell you more about additional options.


