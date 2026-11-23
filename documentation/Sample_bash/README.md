# Legacy nnDoseNet shell templates

These are executable Bash templates for the legacy nnDoseNet workflow. They intentionally do not invoke any other project workflow.

## First use

1. Install the package from the repository root: `python -m pip install -e .`
2. Edit [var.txt](var.txt). Use absolute paths and choose a new, unused `DATASET_ID` for a new experiment.
3. Source the variables in the current shell, then choose a staged action:

```bash
source documentation/Sample_bash/var.txt
bash documentation/Sample_bash/run_legacy_experiment.sh inspect-tr
```

The scripts source `var.txt` themselves, so the explicit `source` command is only useful to inspect variables interactively. `DATASET_NAME="Dicom_Test "` deliberately retains the trailing space used by the source folder. Rename the source folder and change the variable if that name is inconvenient.

## Production stage script

`run_legacy_experiment.sh` provides these deliberate stopping points:

| Action | Runs | Stop and review |
| --- | --- | --- |
| `inspect-tr` | DICOM organization, index, and ROI listing | Review ROI list and create/review `structure.json`. |
| `format-tr` | Stage 1 for training cases | Check images, labels, and beam maps. |
| `preprocess` | Integrity check, plans, preprocessing | Check plans and disk use. |
| `train` | One job per requested fold | Requires enough independent cases and a suitable device. |
| `choose-model` | Cross-validation comparison | Use its generated inference guidance. |
| `format-ts` | Stage 0 + Stage 1 for test cases | Check test channels and mappings. |
| `predict` | Test-set inference | Requires completed fold checkpoints. |
| `evaluate` | Optional DICOM/gamma evaluation | Review NIfTI metrics before clinical use. |

For example:

```bash
bash documentation/Sample_bash/run_legacy_experiment.sh format-tr
bash documentation/Sample_bash/run_legacy_experiment.sh preprocess
CUDA_VISIBLE_DEVICES=0 bash documentation/Sample_bash/run_legacy_experiment.sh train
```

The template uses `--template` beam export by default. Set `BEAM_MODE="bev"` in `var.txt` only when the default BEV projection is intended and sufficient GPU/CPU resources are available.

## Case-specific target prescriptions

For Dicom_Test, set `TARGET_SUMMARY` in your local `var.txt` to an untracked
file that contains the reviewed target prescriptions. Its format is:

```json
{
  "case-folder": {
    "RTSTRUCT ROI name": 36.25
  }
}
```

When `TARGET_SUMMARY` is nonempty, the stage script passes it as `-target_summary`. Stage 1 converts the named target ROIs, writes `Nifti/combined_target.nii.gz`, and uses that as channel `_0001`. Values are `prescription_Gy * 100`; higher prescriptions overwrite lower prescriptions in target overlaps. Set `TARGET_SUMMARY=""` only for a dataset whose Target labels are deliberately defined entirely in `structure.json`.

## Dicom_Test smoke script

`run_dicom_test_smoke.sh` is a reproducible technical validation route, not a training or clinical-evaluation protocol. It does the following:

1. Creates `_handoff_test/legacy` below the repository (an ignored location).
2. Uses hard links when possible so source DICOM is not copied or modified.
3. Runs `Tr` organization and combined Stage 1 with the sample structure mapping and template beams.
4. Runs the mandatory integrity check, creates low-memory plans, and preprocesses 2D data.
5. Runs the same formatting route for `Ts`.

Run it only after confirming `var.txt` points at the source data:

```bash
bash documentation/Sample_bash/run_dicom_test_smoke.sh
```

It stops before training and prediction. Delete its known staging directory with `rm -rf nnDosev2/_handoff_test` when its artifacts are no longer needed. Do not run this script while another run is using that directory. The target-summary file is read from its local source path and is not copied into the repository workspace.

## Sample mapping

[structure_dicom_test_example.json](structure_dicom_test_example.json) is specific to the four-case Dicom_Test prostate subset. Treat it as a starting review artifact, not a generic structure dictionary or clinical protocol.
