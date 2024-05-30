>[!notice]
>Please check **[dataset_inference)](dataset_format_inference.md)** for prediction, Thank you :)
# Evaluation
 nnDoseNet provide sevral evaluation method, there are 2 categories the 1.metrics calculation (numeral) and 2.evaluation map (visual). Metrics calculation is always proceed at the end of trainig.

 ## 1. Metrics Calculation 
 All avalible metrices are calculate contour-wise on both Predciton and GroudTruth(reference). At the end of evaluation, nnDoseNet also average all the metrices for quick assessment.

Avalible metrices for full image:
- `MSE`: the mean-square-error of full image

Avalible metrics for each contour:
- `mean`: the mean dose(Gy) of the contour
- `min`: the min dose(Gy) of the contour
- `max`: the max dose(Gy) of the contour
- `D_0.1_cc`: The min dose received by the hottest sub volume of 0.1cc of each contour

Avalible metrics for only PTV:
- `D_99`: The min dose received by the hottest 99% of voxel of each contour
- `D_95`: The min dose received by the hottest 95% of voxel of each contour
- `D_1`: The min dose received by the hottest 1% of voxel of each contour
- `homogeneity`: The homogeneity of the contour. Calculate by $D_2-D_{98}\over prescription$
- `conformity`: The conformity of the contour. Calculate by $(V_{PTV} \cap V_{prescription})^2\over (V_{PTV}\times V_{prescription})$

Please use this command for metrics calculation

- If you are using image folder of dataset_format(raw): 

        nnDoseNet_evaluate_folder [GroundTruth folder(labelsTs)] [Prediction folder(configuration)] [Image folder(imagesTs)] -djfile [dataset.json path]-id_patient_csv [patientTs.csv path] -sjfile [structure.json path] -np [num of processor]
        
    Since in dataset_format(raw), all contour is convert to a label, we can't seperate the contour if there are 2 contour in same label(e.g. left and right contour). And it's neccessary to put same organ in same label, otherwise augmentation in trainer would mirror the left and right. So seperate the contour based on it's labeled value.\
    This evaluation method would auto-activate in [find_best_configuration](ensemble_model(best_config).md)
    
    To seprate the same labeled contour we need to obtain the image before it's combined, so we need data in nndose_dicom and use below command.

 - If you are using image folder of dataset_format(dicom) :

        nnDoseNet_evaluate_folder [GroundTruth folder(labelsTs)] [Prediction folder(configuration)] [Image folder(Ts)] -djfile [dataset.json path]-id_patient_csv [patientTs.csv path] -sjfile [structure.json path] -np [num of processor] --by_roi

    With `--by_roi` on we evaluate the metrices based on the contour in image folder and structure.json. 

## 2. Evaluation Map


### 2.1 Difference Map

### 2.2.1 Convert Back to DICOM

### 2.2.2 Gamma Analysis