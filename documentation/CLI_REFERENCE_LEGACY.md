# Legacy nnDoseNet command reference

This reference covers every installed `nnDoseNet_*` console command. It intentionally excludes the separate DoseRAD commands.

Run `command -h` before a first production run; the installed CLI is the source of truth for optional arguments. Dataset names are the exact folder names under `nnDose_dicom`, including whitespace. Dataset IDs are normally zero-padded to three digits, for example `990`.

## Shared paths and terms

Source DICOM is under `$nnDose_dicom/<name>/Tr` and `$nnDose_dicom/<name>/Ts`. The five required environment variables are described in [the handoff guide](HANDOFF_NNDOSEV2.md#scope-and-prerequisites).

| Term | Meaning |
| --- | --- |
| `Tr` | Training cases with CT, RTSTRUCT, RTDOSE, and RTPLAN. |
| `Ts` | Held-out cases used for final prediction and optional evaluation. |
| `structure.json` | Reviewed mapping from source ROI names to Target, OAR, and Body masks. |
| `DatasetNNN_Name` | Dataset folder created from numeric ID `NNN` and dataset name. |
| plans identifier | The generated plan file name without `.json`, normally `nnDosePlans`. Use the same value for preprocessing, training, and prediction. |

## Stage 0 — inspect DICOM and review structures

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_0-0_list_dicom_and_organize -name NAME -trts Tr|Ts` | Sorts incoming DICOM by modality and makes a plan-centric index. This changes the source tree by creating `Dicom/CT`, `Dicom/RS`, `Dicom/RD`, and `Dicom/RP`. | `$nnDose_dicom/NAME/Tr_dicom_index.csv` or `Ts_dicom_index.csv` |
| `nnDoseNet_0-1_list_roi -name NAME -trts Tr|Ts` | Reads RTSTRUCT ROI names, excluding setup markers. | `Tr_roi_list_summary.csv` and `.json`, or `Ts_...` |
| `nnDoseNet_0-2a_suggest_structure_csv -name NAME -trts Tr|Ts` | Matches listed ROIs to the organ dictionary. Review this output rather than accepting it blindly. | `suggest_structure.csv` |
| `nnDoseNet_0-2b_csv_to_structure_json -name NAME [-csv FILE]` | Converts the reviewed suggestion CSV to a mapping. | `structure_organ.json` |
| `nnDoseNet_0-2_suggest_structure -name NAME -trts Tr|Ts` | Runs the preceding suggestion/conversion route in one command. | `structure_organ.json` |
| `nnDoseNet_0-3_list_target -name NAME -trts Tr|Ts [--prescription VALUE]` | Lists target ROIs and writes target/prescription metadata. `VALUE` can be a clinical Gy value or `D90`, `D95`, or `D98`. | `structure_target.json` |

`D90`/`D95`/`D98` derives an estimate from the delivered dose. It is not a substitute for reviewing the clinical prescription.

## Stage 1 — create the NIfTI dataset

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_1-1_convert_dicom_image -name NAME -trts Tr|Ts -structure FILE [-target_summary FILE]` | Converts CT, dose, and configured structures to per-case NIfTI. RTDOSE is resampled onto the CT grid. With a target summary, creates `combined_target.nii.gz` from case-specific prescriptions. | `<case>/Nifti/*.nii.gz` |
| `nnDoseNet_1-2_label_nifti -name NAME -id NNN -trts Tr|Ts -structure FILE [-target_summary FILE]` | Combines configured masks into Target/OAR/Body channels and creates the raw training layout or formatted test layout. With a target summary, Target preserves `combined_target` prescription values. | `imagesTr`/`labelsTr` or `GroundTruth/imagesTs`/`labelsTs`, plus patient-ID mapping CSV |
| `nnDoseNet_1-3_beam_export -name NAME -trts Tr|Ts -structure FILE [--template]` | Exports `beam.nii.gz` from RTPLAN. `--template` is quick and deterministic; default BEV projection is expensive on CPU. | `<case>/Nifti/beam.nii.gz` and the raw/test beam channel |
| `nnDoseNet_1_format_dataset ...` | Runs conversion, labeling, and beam export in the same order. It is the recommended normal entry point. | All Stage-1 outputs |
| `nnDoseNet_1-1_organize_and_convert_dicom` | Backwards-compatible alias for `1-1_convert_dicom_image`. | Same as conversion |
| `nnDoseNet_1-2_label_nifti_and_final_formatting` | Backwards-compatible alias for `1-2_label_nifti`. | Same as labeling |

The combined command accepts `--organized_dicom` when Stage 0 has already organized input. Pass the exact same absolute `-structure` path to conversion, labeling, and beam export. The output image channels are CT (`_0000`), Target (`_0001`), OAR (`_0002`), and Body (`_0003`); the label is dose.

Use `-target_summary` when targets and prescriptions are case-specific. The JSON maps each case-folder name to `{RTSTRUCT_ROI_name: prescription_Gy}`. Stage 1 converts those named ROIs, writes `combined_target.nii.gz` using Gy × 100, and resolves overlaps in favor of the higher prescription.

## Stage 2 — validate, plan, and preprocess

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_2-1_extract_fingerprint -d NNN --verify_dataset_integrity` | Checks all channels and dose labels, measures the training set, and makes the dataset fingerprint. Use `--clean` after changing Stage-1 data. | `$nnDose_preprocessed/DatasetNNN_Name/dataset_fingerprint.json` |
| `nnDoseNet_2-2_plan_experiment -d NNN` | Chooses network configurations, patch sizes, normalizers, and batch sizes from the fingerprint. | `nnDosePlans.json` (or custom plans name) |
| `nnDoseNet_2-3_preprocess -d NNN -plans_name NAME -c CONFIG ...` | Resamples/normalizes cases and writes training arrays for one or more configurations. | Configuration-specific `.npz`/metadata under the preprocessed dataset |
| `nnDoseNet_2_plan_and_preprocess -d NNN ...` | Runs fingerprinting, planning, then preprocessing. | All Stage-2 outputs |

The integrity check is a hard gate: do not plan or train until it passes. A custom `-gpu_memory_target` requires a custom `-overwrite_plans_name`; carry that plans name to later commands.

## Stage 3 — cross-validation training and model selection

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_3_train DATASET CONFIG FOLD -p PLANS -device cuda` | Trains one fold, writes checkpoints, then produces the fold validation prediction. `FOLD` is normally `0` through `4`. | `$nnDose_results/DatasetNNN_Name/nnDoseTrainer__PLANS__CONFIG/fold_F/` |
| `nnDoseNet_3_find_best_configuration DATASET -p PLANS -c CONFIGS -f 0 1 2 3 4` | Compares completed cross-validation outputs, optionally builds ensembles, and writes inference instructions. | `inference_information.json` in the results area |

Use five folds only with at least five independent training cases. The `-device cpu` option is useful for debugging, but not a practical production training path for full datasets. The output of `find_best_configuration` determines whether to predict a single configuration or an ensemble.

## Stage 4 — inference and result aggregation

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_4_predict -i INPUT -o OUTPUT -d DATASET -c CONFIG -f ...` | Predicts arbitrary correctly-channelled NIfTI cases from a trained model. | One predicted dose NIfTI per input case in `OUTPUT` |
| `nnDoseNet_4_predict_Ts -id NNN -name NAME -c CONFIG -p PLANS -f ...` | Predicts the Stage-1 formatted test set. It also writes a dose/DVH summary when test labels are present. | `$nnDose_prediction/NAME/DatasetNNN_Name/Prediction/CONFIG/` and `summary.json` |
| `nnDoseNet_4_accumulate_crossval_results DATASET -c CONFIG -f ...` | Collects validation predictions from the requested folds and evaluates the joint folder. | `crossval_results_folds_*` in the trained-model folder unless `-o` is supplied |
| `nnDoseNet_4_ensemble -i FOLDER ... -o OUTPUT` | Averages compatible prediction folders. | Ensembled NIfTI predictions plus copied plans/dataset metadata |

For `predict` and `predict_Ts`, a requested fold must have a valid final checkpoint. `-f` can name individual folds or `all`; it must match what was trained. Use the model-selection output when an ensemble is recommended.

## Stage 5 — metrics, DICOM export, and visual QA

| Command | What it does | Primary output |
| --- | --- | --- |
| `nnDoseNet_5_evaluate_folder GT PRED INPUT -djfile DATASET_JSON [-sjfile STRUCTURE]` | Computes dose and DVH metrics, optionally by original ROI contour. | `PRED/summary.json` by default, or `-o FILE` |
| `nnDoseNet_5_dicom_based_evaluation -pred PRED -ref DICOM_TS -csv PATIENT_MAP` | Converts predicted NIfTI dose back to DICOM and runs gamma analysis unless disabled. | DICOM prediction folder and `gamma_summary.json` |
| `nnDoseNet_5_dicom_based_evaluation_Ts -id NNN -name NAME` | Runs the preceding conversion/evaluation with Stage-1 test paths. | Results under the test prediction tree |
| `nnDoseNet_5_diff_map_evaluation -gt_folder GT -pred_folder PRED` | Writes per-case voxelwise difference maps for visual review. | `PRED/DiffMap/` by default |

Gamma analysis can consume substantial RAM. Reduce `-np` for this command. A DICOM export is only meaningful after the corresponding NIfTI prediction has passed clinical and geometry review.

## Installed command inventory

`pyproject.toml` is the only TOML file in this repository. Its `[project.scripts]` table installs the 26 legacy commands listed above; it does not configure a cohort or experiment. Cohort-specific settings reside in `structure.json`, the generated `dataset.json`, the fingerprint, plans, and split files described in the [handoff guide](HANDOFF_NNDOSEV2.md#configuration-files).
