>[!notice]
>Please check **[dataset_inference)](dataset_format_inference.md)** for prediction, Thank you :)
# Evaluation
 nnDoseNet provide sevral evaluation method, there are 2 categories the 1.metrics calculation (numeral) and 2.evaluation map (visual). Metrics calculation is always proceed at the end of trainig.

 ## 1. Metrics Calculation 
 All avalible metrices are calculate contour-wise on both Predciton and GroudTruth(reference). At the end of evaluation, nnDoseNet also average all the metrices for quick assessment.

Avalible metrices of full image:
 - `MSE`: the mean-square-error of full image

Avalible metrics in each contour:
 - `mean`: the mean dose(Gy) of the contour
 - `min`: the min dose(Gy) of the contour
 - `max`: the max dose(Gy) of the contour
 - `D_0.1_cc`: The min dose received by the hottest sub volume of 0.1cc of each contour
 - `D_99`: The min dose received by the hottest 99% of voxel of each contour
 - `D_95`: The min dose received by the hottest 95% of voxel of each contour
 - `D_1`: The min dose received by the hottest 1% of voxel of each contour
 - `homogeneity`: The homogeneity of the dose in contour. Calculate by (D_2-D_98)/D_50



## 2. Evaluation Map

### 2.1 Difference Map

### 2.2.1 Convert Back to DICOM

### 2.2.2 Gamma Analysis