#!/usr/bin/env bash
# Run one reviewed stage of the legacy nnDoseNet workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=var.txt
source "${SCRIPT_DIR}/var.txt"

export nnDose_dicom nnDose_raw nnDose_preprocessed nnDose_results nnDose_prediction
cd "${NNDOSE_REPO}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 2
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command is not installed: $1" >&2
        echo "Run: cd ${NNDOSE_REPO} && python -m pip install -e ." >&2
        exit 2
    fi
}

for required_command in \
    nnDoseNet_0-0_list_dicom_and_organize \
    nnDoseNet_1_format_dataset \
    nnDoseNet_2-1_extract_fingerprint \
    nnDoseNet_3_train; do
    require_command "${required_command}"
done

format_args=(
    -name "${DATASET_NAME}"
    -id "${DATASET_ID}"
    -nps "${N_PROCESSES}"
    -structure "${STRUCTURE_JSON}"
)

if [[ -n "${TARGET_SUMMARY:-}" ]]; then
    require_file "${TARGET_SUMMARY}"
    format_args+=(-target_summary "${TARGET_SUMMARY}")
fi

case "${BEAM_MODE}" in
    template)
        format_args+=(--template)
        ;;
    bev)
        ;;
    *)
        echo "BEAM_MODE must be template or bev; received: ${BEAM_MODE}" >&2
        exit 2
        ;;
esac

action="${1:-}"
case "${action}" in
    inspect-tr)
        nnDoseNet_0-0_list_dicom_and_organize -name "${DATASET_NAME}" -trts Tr -nps "${N_PROCESSES}"
        nnDoseNet_0-1_list_roi -name "${DATASET_NAME}" -trts Tr -nps "${N_PROCESSES}" --organized_dicom
        nnDoseNet_0-2a_suggest_structure_csv -name "${DATASET_NAME}" -trts Tr
        nnDoseNet_0-3_list_target -name "${DATASET_NAME}" -trts Tr
        echo "Review the ROI list and structure mapping before format-tr."
        ;;
    format-tr)
        require_file "${STRUCTURE_JSON}"
        nnDoseNet_1_format_dataset "${format_args[@]}" -trts Tr --organized_dicom
        ;;
    preprocess)
        nnDoseNet_2-1_extract_fingerprint -d "${DATASET_ID}" -np "${PREPROCESS_PROCESSES}" --verify_dataset_integrity --clean
        nnDoseNet_2-2_plan_experiment -d "${DATASET_ID}" -overwrite_plans_name "${PLANS_NAME}"
        nnDoseNet_2-3_preprocess -d "${DATASET_ID}" -plans_name "${PLANS_NAME}" -c "${TRAIN_CONFIGURATION}" -np "${PREPROCESS_PROCESSES}"
        ;;
    train)
        for fold in "${FOLDS[@]}"; do
            nnDoseNet_3_train "${DATASET_ID}" "${TRAIN_CONFIGURATION}" "${fold}" \
                -tr "${TRAINER}" -p "${PLANS_NAME}" -device "${DEVICE}"
        done
        ;;
    choose-model)
        nnDoseNet_3_find_best_configuration "${DATASET_ID}" \
            -tr "${TRAINER}" -p "${PLANS_NAME}" -c "${TRAIN_CONFIGURATION}" -f "${FOLDS[@]}"
        ;;
    format-ts)
        require_file "${STRUCTURE_JSON}"
        nnDoseNet_0-0_list_dicom_and_organize -name "${DATASET_NAME}" -trts Ts -nps "${N_PROCESSES}"
        nnDoseNet_1_format_dataset "${format_args[@]}" -trts Ts --organized_dicom
        ;;
    predict)
        nnDoseNet_4_predict_Ts -id "${DATASET_ID}" -name "${DATASET_NAME}" \
            -tr "${TRAINER}" -p "${PLANS_NAME}" -c "${TRAIN_CONFIGURATION}" \
            -f "${FOLDS[@]}" -device "${DEVICE}" -npp "${N_PROCESSES}" -nps "${N_PROCESSES}"
        ;;
    evaluate)
        nnDoseNet_5_dicom_based_evaluation_Ts -id "${DATASET_ID}" -name "${DATASET_NAME}" \
            -c "${TRAIN_CONFIGURATION}" -np "${N_PROCESSES}"
        ;;
    *)
        echo "Usage: $0 {inspect-tr|format-tr|preprocess|train|choose-model|format-ts|predict|evaluate}" >&2
        exit 2
        ;;
esac
