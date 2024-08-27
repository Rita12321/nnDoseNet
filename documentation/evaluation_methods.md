>[!notice]
>Please check **[dataset_inference)](dataset_format_inference.md)** for prediction, Thank you :)
# Evaluation
 nnDoseNet provide several evaluation method, there are 2 categories the 1.metrics calculation (numeral) and 2.evaluation map (visual). Metrics calculation is always proceed at the end of training.

 ## 1. Metrics Calculation 
 All available metrics are calculate contour-wise on both Prediction and GroundTruth(reference). At the end of evaluation, nnDoseNet also average all the metrics for quick assessment.

Available metrics for full image:
- `MSE`: The mean-square-error of full image
- `max`: The max dose of the entire dose map
- `# < 105% prescription`: The number of cases in this run of analysis that has max dose of entire dose map less than 105% of prescription.
- `# < 107% prescription`: The number of cases in this run of analysis that has max dose of entire dose map less than 107% of prescription.

Available metrics for each contour:
- `mean`: The mean dose(Gy) of the contour
- `min`: The min dose(Gy) of the contour
- `max`: The max dose(Gy) of the contour
- `D_0.1_cc`: The min dose received by the hottest sub volume of 0.1cc of each contour

Available metrics for only PTV:
- `D_99`: The min dose received by the hottest 99% of voxel of each contour
- `D_95`: The min dose received by the hottest 95% of voxel of each contour
- `D_1`: The min dose received by the hottest 1% of voxel of each contour
- `homogeneity`: The homogeneity of the contour. Calculate by $D_2-D_{98}\over prescription$
- `conformity`: The conformity of the contour. Calculate by $(V_{PTV} \cap V_{prescription})^2\over (V_{PTV}\times V_{prescription})$
- `# > target prescription of D_99`: The number of cases in this run of analysis that has D_99 greater than prescription.
- `# > target prescription of D_95`: The number of cases in this run of analysis that has D_95 greater than prescription.

*Note: nnDoseNet don't normalize the dose.

Please use this command for metrics calculation

- If you are using image folder of dataset_format(raw): 

        nnDoseNet_evaluate_folder [GroundTruth folder(labelsTs)] [Prediction folder(configuration)] [Image folder(imagesTs)] -djfile [dataset.json path]-id_patient_csv [patientTs.csv path] -sjfile [structure.json path] -np [num of processor]
        
    Since in dataset_format(raw), all contour is convert to a label, we can't separate the contour if there are 2 contour in same label(e.g. left and right contour). And it's necessary to put same organ in same label, otherwise augmentation in trainer would mirror the left and right. So separate the contour based on it's labeled value.\
    This evaluation method would auto-activate in [find_best_configuration](ensemble_model(best_config).md)
    
    To separate the same labeled contour we need to obtain the image before it's combined, so we need data in nndose_dicom and use below command.

 - If you are using image folder of dataset_format(dicom) :

        nnDoseNet_evaluate_folder [GroundTruth folder(labelsTs)] [Prediction folder(configuration)] [Image folder(Ts)] -djfile [dataset.json path]-id_patient_csv [patientTs.csv path] -sjfile [structure.json path] -np [num of processor] --by_roi

    With `--by_roi` on we evaluate the metrics based on the contour in image folder and structure.json. 

## 2. Evaluation Map
 Evaluation map are consist of Nifti based and DICOM based. For [Nifti based dataset](dataset_format(nnDose_dicom).md) nnDoseNet only provide difference map for visualized evaluation. For [Dicom based dataset](dataset_format(nnDose_dicom).md), nnDoseNet provide a function to convert the prediction Nifti files into a dcm for  further investigation. With dcm format of prediction we use gamma analysis function from [pymedphys](https://github.com/pymedphys/pymedphys). 

### 2.1 Difference Map
Please use this command to generate difference map
    
    nnDoseNet_diff_map_evaluation -gt_folder [groundtruth folder(labelsTs)] -pred_folder [prediction folder(config)] -np [num of processor].

### 2.2.1 Convert Back to DICOM and Gamma Analysis
Please use this command to convert prediction to DICOM format and proceed gamma analysis.

    nnDoseNet_dicom_based_evaluation -gt_folder [groundtruth folder(labelsTs)] -pred_folder [prediction folder(config)] -np [num of processor]

Please use this command to convert prediction to DICOM format and DON'T want proceed gamma analysis.\
*Strongly suggest to put `--no_gamma` tag if you don't need  gamma analysis.

    nnDoseNet_dicom_based_evaluation -gt_folder [groundtruth folder(labelsTs)] -pred_folder [prediction folder(config)] -np [num of processor] --no_gamma

