# Dicom_Test validation record

## Scope

This record covers the legacy nnDoseNet workflow only. It was run against a temporary hard-linked working copy of the source folder named `Dicom_Test ` under `nnDose_1_dicom`. The trailing space is part of the original directory name, so command variables must preserve it or use a renamed copy.

The copy contained four readable RT cases:

| Partition | Cases | DICOM inventory per case |
| --- | --- | --- |
| `Tr` | 2 de-identified validation cases | CT, RTSTRUCT, RTDOSE, RTPLAN |
| `Ts` | 2 de-identified validation cases | CT, RTSTRUCT, RTDOSE, RTPLAN |

All source DICOM objects were readable with pydicom before processing.

## Mapping used for this smoke test

The test used [structure_dicom_test_example.json](Sample_bash/structure_dicom_test_example.json) for OAR/Body and source ROI conversion. The real run used `Prostate_target_summary.json` for case-specific targets and prescriptions.

This is a technical test mapping, not a clinical standard. Review target definitions, laterality, and prescriptions with the treating team before reusing it. With `-target_summary`, Stage 1 creates `combined_target.nii.gz` and preserves prescription values as Gy × 100 in channel `_0001`.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| DICOM organization and plan index (`Tr`) | Passed | Two plan-index rows created. |
| ROI listing and structure review inputs (`Tr`) | Passed | ROI lists created; observed Body, target, bladder, rectum, femur, and bowel structures. |
| CT/dose/structure conversion (`Tr`) | Passed | Per-case NIfTI written. RTDOSE was resampled to the CT grid. |
| Target/OAR/Body labeling (`Tr`) | Passed | All expected image channels were written. |
| Template beam export (`Tr`) | Passed | Beam maps produced for both training cases. |
| Dataset integrity/fingerprint | Passed | CT, all four input channels, and dose labels shared a grid after resampling. |
| Planning | Passed | `HandoffPlans` created: 2D, 3D low-resolution, and 3D full-resolution configurations. |
| 2D preprocessing | Passed | Two compressed preprocessed cases written. |
| One CPU training step | Passed | Forward pass, dose/DVH loss, backward pass, optimizer step, and nonzero parameter update completed. |
| Complete test formatting (`Ts`) | Passed | Two cases formatted with template beams. Both had nonzero Target/OAR/Body masks and dose grid matching CT. |
| Target-summary integration | Passed | Case-specific target prescriptions were preserved in all formatted validation cases. |

The final test-case channel check confirmed nonempty Target, OAR, and Body masks and a dose grid matching CT for both test cases. Exact case identifiers, image dimensions, voxel counts, and prescription values are intentionally kept out of this repository.

## Corrections made during validation

The checks exposed and corrected the following legacy-pipeline defects:

1. Label formatting now writes Target (`_0001`), OAR (`_0002`), and Body (`_0003`) masks. A body-support fallback is emitted when a configured body contour is absent.
2. Generated `dataset.json` now declares channel 3 as `BODY` and includes all three mask channels, matching the planner and trainer.
3. RTDOSE is resampled to the CT grid during conversion; this makes the Stage-2 geometry check meaningful and passable.
4. Stage-1 beam export receives the explicit `-structure` mapping supplied to the combined formatting command.
5. Formatted case identifiers are stable anonymous sequence IDs based on sorted case folders, rather than DICOM patient IDs.
6. CPU training now performs the backward pass when mixed-precision gradient scaling is unavailable.
7. Stage 1 now accepts `-target_summary`, creates patient-specific `combined_target.nii.gz` from clinical prescription data, and uses it for the Target channel. Higher prescription targets take precedence in overlaps.
8. Dataset/prediction output names trim incidental whitespace from the source DICOM dataset directory. This prevents a folder such as `Dicom_Test ` from creating an incompatible `Dataset990_Dicom_Test ` output name.

## Not claimed by this smoke test

This subset has only two training cases and the validation host has no CUDA device. Therefore it is **not** a valid five-fold model-development or clinical-performance study.

- Default BEV beam projection was started successfully but was stopped because full CPU projection was impractically slow. Template beam export passed.
- Five-fold production training, best-configuration selection, full test-set inference, ensembling, and final gamma/DICOM evaluation were not run. They require a trained model and GPU-capable validation environment.
- A `D95` target estimate was evaluated only as a code path. Its value was not accepted as a clinical prescription; use reviewed clinical prescription data.

## Reproducibility route

After reviewing the mapping, use the [Dicom_Test smoke script](Sample_bash/README.md#dicom_test-smoke-script). It creates an ignored staging workspace, keeps the original source untouched, and exercises `Tr` and `Ts` formatting plus the Stage-2 integrity check. It deliberately stops before training and prediction.
