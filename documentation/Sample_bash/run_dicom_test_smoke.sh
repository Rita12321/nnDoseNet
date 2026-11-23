#!/usr/bin/env bash
# Exercise DICOM -> NIfTI -> raw -> integrity -> plans -> 2D preprocessing for Dicom_Test.
# This is a technical smoke test. It intentionally does not train or predict.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=var.txt
source "${SCRIPT_DIR}/var.txt"

export nnDose_dicom nnDose_raw nnDose_preprocessed nnDose_results nnDose_prediction
cd "${NNDOSE_REPO}"

SOURCE_DATASET="${nnDose_dicom}/${DATASET_NAME}"
TEST_ROOT="${NNDOSE_REPO}/_handoff_test/legacy"
TEST_DICOM="${TEST_ROOT}/dicom"
TEST_RAW="${TEST_ROOT}/raw"
TEST_PREPROCESSED="${TEST_ROOT}/preprocessed"
TEST_RESULTS="${TEST_ROOT}/results"
TEST_PREDICTION="${TEST_ROOT}/prediction"
TEST_NAME="Dicom_Test"
TEST_ID="990"
TEST_STRUCTURE="${TEST_DICOM}/${TEST_NAME}/structure_handoff_test.json"

if [[ ! -d "${SOURCE_DATASET}" ]]; then
    echo "Source dataset not found: ${SOURCE_DATASET}" >&2
    exit 2
fi
if [[ -e "${TEST_ROOT}" ]]; then
    echo "Staging directory already exists: ${TEST_ROOT}" >&2
    echo "Review it or remove that exact directory before rerunning." >&2
    exit 2
fi

mkdir -p "${TEST_DICOM}" "${TEST_RAW}" "${TEST_PREPROCESSED}" "${TEST_RESULTS}" "${TEST_PREDICTION}"
# The staging location is within the project filesystem so hard links avoid copying DICOM bytes.
cp -al "${SOURCE_DATASET}" "${TEST_DICOM}/${TEST_NAME}"
cp "${SCRIPT_DIR}/structure_dicom_test_example.json" "${TEST_STRUCTURE}"

format_args=(--organized_dicom --template -structure "${TEST_STRUCTURE}")
if [[ -n "${TARGET_SUMMARY:-}" ]]; then
    if [[ ! -f "${TARGET_SUMMARY}" ]]; then
        echo "Target summary not found: ${TARGET_SUMMARY}" >&2
        exit 2
    fi
    # Keep the source prescription summary outside the staging workspace.
    format_args+=(-target_summary "${TARGET_SUMMARY}")
fi

export nnDose_dicom="${TEST_DICOM}"
export nnDose_raw="${TEST_RAW}"
export nnDose_preprocessed="${TEST_PREPROCESSED}"
export nnDose_results="${TEST_RESULTS}"
export nnDose_prediction="${TEST_PREDICTION}"

nnDoseNet_0-0_list_dicom_and_organize -name "${TEST_NAME}" -trts Tr -nps 1
nnDoseNet_0-1_list_roi -name "${TEST_NAME}" -trts Tr -nps 1 --organized_dicom
nnDoseNet_1_format_dataset -name "${TEST_NAME}" -id "${TEST_ID}" -trts Tr -nps 1 \
    "${format_args[@]}"

nnDoseNet_2-1_extract_fingerprint -d "${TEST_ID}" -np 1 --verify_dataset_integrity --clean --verbose
nnDoseNet_2-2_plan_experiment -d "${TEST_ID}" -gpu_memory_target 1 -overwrite_plans_name HandoffPlans
nnDoseNet_2-3_preprocess -d "${TEST_ID}" -plans_name HandoffPlans -c 2d -np 1 --verbose

nnDoseNet_0-0_list_dicom_and_organize -name "${TEST_NAME}" -trts Ts -nps 1
nnDoseNet_1_format_dataset -name "${TEST_NAME}" -id "${TEST_ID}" -trts Ts -nps 1 \
    "${format_args[@]}"

echo "Smoke test completed. Staging artifacts: ${TEST_ROOT}"
echo "It deliberately stopped before training and prediction."
