# Legacy nnDoseNet v2 workflow

## Scope and prerequisites

This guide covers the original `nnDoseNet_*` pipeline. It does not describe the separate DoseRAD workflow.

Install the repository in the intended Python environment, then set all five paths before invoking a CLI:

```bash
cd /path/to/nnDosev2
python -m pip install -e .
source documentation/Sample_bash/var.txt
```

The variables must be absolute, writable paths:

| Variable | Purpose |
| --- | --- |
| `nnDose_dicom` | Source DICOM/NIfTI cases, divided into `Tr` and `Ts` |
| `nnDose_raw` | Labeled training dataset, `DatasetNNN_Name` |
| `nnDose_preprocessed` | Fingerprints, plans, and preprocessed arrays |
| `nnDose_results` | Trained folds, checkpoints, validation results, best-model instructions |
| `nnDose_prediction` | Formatted test data, model predictions, metrics, and DICOM exports |

The DICOM organization stage moves DICOM files into `Dicom/CT`, `Dicom/RS`, `Dicom/RD`, and `Dicom/RP` below each case. Run it on a copy when the incoming data must remain byte-for-byte unchanged.

The `-name` value must match the source DICOM folder exactly, including any whitespace. Generated `DatasetNNN_Name` and prediction-directory names trim accidental leading/trailing whitespace so that a source folder such as `Dicom_Test ` produces `DatasetNNN_Dicom_Test`.

## Data flow and outputs

| Stage | Required input | Main command | Output |
| --- | --- | --- | --- |
| 0: inspect | Unorganized or organized DICOM | `nnDoseNet_0-*` | DICOM index, ROI list, editable structure suggestions, target summary |
| 1: format training | `Tr` DICOM and reviewed `structure.json` | `nnDoseNet_1_format_dataset ... -trts Tr` | `DatasetNNN_Name/imagesTr`, `labelsTr`, `dataset.json`, case mapping |
| 2: plan/preprocess | Formatted training dataset | `nnDoseNet_2_plan_and_preprocess` | fingerprint, `nnDosePlans.json`, configuration-specific `.npz` files |
| 3: train | Preprocessed data and a GPU | `nnDoseNet_3_train` for each fold | checkpoints and each fold's `validation/` output |
| 3: choose model | Completed folds | `nnDoseNet_3_find_best_configuration` | `inference_information.json` and exact recommended inference command |
| 1: format test | `Ts` DICOM and the same reviewed structure file | `nnDoseNet_1_format_dataset ... -trts Ts` | `nnDose_prediction/Name/DatasetNNN_Name/GroundTruth/imagesTs` and `labelsTs` |
| 4: predict | Formatted test data and a completed model | `nnDoseNet_4_predict_Ts` | `Prediction/<configuration>/*.nii.gz` plus `summary.json` |
| 5: evaluate | Prediction, test dose labels, input masks | `nnDoseNet_5_evaluate_folder` | per-case and aggregate dose/DVH metrics CSV/JSON |

The generated `dataset.json` uses four input channels:

| Suffix | Content |
| --- | --- |
| `_0000` | CT (or declared primary image modality) |
| `_0001` | Target masks, or prescription-valued `combined_target` when `-target_summary` is used (Gy × 100) |
| `_0002` | OAR masks |
| `_0003` | Body/dose-support mask |

All four channels and the dose label must be on exactly the same image grid. Stage 2 with `--verify_dataset_integrity` is the mandatory gate that checks this before planning.

## Required review points

1. Run `nnDoseNet_0-1_list_roi` and inspect the generated ROI list.
2. Create or edit `structure.json`. Every name in `structure_label` must be a final key from `structure_name`.
3. Run `nnDoseNet_0-3_list_target` with clinical prescriptions when available. Do not substitute D95/D90-derived values for the clinical prescription without review.
4. Format `Tr`, then run the Stage-2 integrity check before spending GPU time.
5. Train all five folds only when at least five training cases exist. Use a manual split file for a deliberate alternate split.
6. Format `Ts` with the same structure mapping and run the configuration/folds selected by `nnDoseNet_3_find_best_configuration`.

## Configuration files

`pyproject.toml` is the repository's only TOML file. It defines Python dependencies and the installed `nnDoseNet_*` console commands; it does not hold patient, training, or experiment parameters.

Runtime configuration is instead split across these generated or user-reviewed files:

| File | Owner | Role |
| --- | --- | --- |
| `nnDose_dicom/<name>/structure.json` | User | Maps source ROI names to final masks and Target/OAR/Body label groups |
| `nnDose_dicom/<name>/*_target_summary.json` | User | Optional case-specific target ROI-to-prescription mapping (Gy). Passed to Stage 1 with `-target_summary`; it creates `combined_target.nii.gz` using Gy × 100 and higher-dose overlap precedence. |
| `nnDose_raw/DatasetNNN_Name/dataset.json` | Stage 1 | Declares the four image channels and dose label format |
| `nnDose_preprocessed/DatasetNNN_Name/dataset_fingerprint.json` | Stage 2 | Records geometry/intensity properties measured from training data |
| `nnDose_preprocessed/DatasetNNN_Name/nnDosePlans.json` | Stage 2 | Contains planned 2D/3D network, patch, preprocessing, and loss settings |
| `nnDose_preprocessed/DatasetNNN_Name/splits_final.json` | Training | Stores the five-fold split; review before any production run |

## Beam-map modes

Stage 1 can make `beam.nii.gz` from the RTPLAN.

- `--template` is deterministic and quick. It uses plan index metadata and the supplied templates.
- The default BEV mode projects fields/control points from the DICOM plan and is much more computationally expensive on CPU. Use `--gpu` where CUDA is available.

Pass the same `-structure /absolute/path/structure.json` to the Stage-1 combined command or to `nnDoseNet_1-3_beam_export`; it is used to identify target masks for beam export.

When clinical prescriptions vary by case, pass `-target_summary /absolute/path/target_summary.json` to `nnDoseNet_1_format_dataset` (or both separate conversion and labeling commands). The summary has the form `{ "case-folder": { "RTSTRUCT ROI name": prescription_Gy } }`. Stage 1 creates `Nifti/combined_target.nii.gz`; its Target channel preserves prescription values as Gy × 100. The shared `structure.json` still supplies OAR and Body masks.

## Running an experiment

The templates in [Sample_bash](Sample_bash/README.md) safely separate each stage. A typical production order is:

```bash
bash documentation/Sample_bash/run_legacy_experiment.sh inspect-tr
# Review ROI output and structure.json here.
bash documentation/Sample_bash/run_legacy_experiment.sh format-tr
bash documentation/Sample_bash/run_legacy_experiment.sh preprocess
bash documentation/Sample_bash/run_legacy_experiment.sh train
bash documentation/Sample_bash/run_legacy_experiment.sh choose-model
bash documentation/Sample_bash/run_legacy_experiment.sh format-ts
bash documentation/Sample_bash/run_legacy_experiment.sh predict
bash documentation/Sample_bash/run_legacy_experiment.sh evaluate
```

Use `nnDoseNet_3_find_best_configuration`'s generated instructions for the final prediction command when it selects an ensemble rather than one configuration.
