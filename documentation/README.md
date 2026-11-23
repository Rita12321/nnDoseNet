# nnDoseNet v2 handoff guide

This directory documents the legacy nnDoseNet v2 workflow only:

```text
DICOM (Tr/Ts) → NIfTI + masks → raw dataset → fingerprint/plans → preprocessed data
→ cross-validation training → test prediction → dose/DVH/DICOM evaluation
```

Start with [Handoff workflow](HANDOFF_NNDOSEV2.md). It explains the required paths, safe order of operations, and the distinction between a smoke test and a full training run.

Use [CLI and configuration reference](CLI_REFERENCE_LEGACY.md) to look up every installed `nnDoseNet_*` command, its inputs, and where it writes output.

[Dicom_Test validation](DICOM_TEST_VALIDATION.md) records the end-to-end checks performed against the included prostate DICOM subset and the limits of that subset.

Runnable templates are in [Sample_bash](Sample_bash/README.md). Copy and edit `var.txt` before running the scripts.

The existing format and installation pages remain useful references:

- [DICOM dataset format](dataset_format(nnDose_dicom).md)
- [Raw NIfTI dataset format](dataset_format(nnDose_raw).md)
- [Environment paths](setting_up_paths.md)
- [Evaluation methods](evaluation_methods.md)
