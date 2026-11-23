from nndosenet.configuration import default_num_processes
from nndosenet.formating_dataset.format_dataset_api import (
    format_dataset_from_dicom, format_dataset_from_nifti,
    organize_dicom, organize_nifti,
    list_roi_names_from_dicom, suggest_structure_from_roi_list,
    csv_to_structure_organ_json, write_target_summary_json, list_dicom_index,
    beam_export,
)
from nndosenet.paths import nnDose_raw, nnDose_dicom
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
import os
import json


# =============================================================================
# Stage 0 — DICOM Organization & Structure Assistance
# =============================================================================

def list_dicom_entry():
    """nnDoseNet_0-0_list_dicom_and_organize: Organize DICOM + build plan-centric index."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Organize raw DICOM files by modality (CT/RS/RD/RP) and build a plan-centric DICOM index CSV."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom. Example: Prostate")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=3,
                        help="Number of processes for organizing DICOM files")
    parser.add_argument('--organized_dicom', default=False, action="store_true",
                        help="Skip DICOM organization (files already in CT/RS/RD/RP folders)")
    parser.add_argument('--continue', dest='continue_flag', default=False, action="store_true",
                        help="Skip patients that already have organized DICOM folders")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    if not args.organized_dicom:
        organize_dicom(join(raw_dicom_folder, args.trts), num_processes=args.nps)
    list_dicom_index(raw_dicom_folder, tr_ts=args.trts)


def list_roi_entry():
    """nnDoseNet_0-1_list_roi: List ROI names from RT Struct DICOM per patient."""
    import argparse
    parser = argparse.ArgumentParser(
        description="List ROI structure names from RT Struct DICOM, excluding setup markers."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=1,
                        help="Number of processes")
    parser.add_argument('--organized_dicom', default=False, action="store_true",
                        help="Skip DICOM organization")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    if not args.organized_dicom:
        organize_dicom(join(raw_dicom_folder, args.trts))
    list_roi_names_from_dicom(raw_dicom_folder, tr_ts=args.trts, num_processes=args.nps)


def suggest_structure_csv_entry():
    """nnDoseNet_0-2a_suggest_structure_csv: Match ROI names to canonical organs, output editable CSV."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Match ROI names against organ dictionary and output suggest_structure.csv for user editing."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-organ_dict', type=str, default=None,
                        help="Path to organ_dictionary.json (default: bundled)")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    roi_json = join(raw_dicom_folder, args.trts + "_roi_list_summary.json")
    if not os.path.isfile(roi_json):
        print("ROI list not found at %s. Run nnDoseNet_0-1_list_roi first." % roi_json)
        return
    suggest_structure_from_roi_list(
        roi_json, organ_dictionary_path=args.organ_dict,
        output_path=raw_dicom_folder, output_csv_only=True,
    )


def csv_to_structure_json_entry():
    """nnDoseNet_0-2b_csv_to_structure_json: Convert edited CSV to structure_organ.json."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert user-edited suggest_structure.csv to structure_organ.json."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-csv', type=str, default=None,
                        help="Path to suggest_structure.csv (default: nnDose_dicom/{name}/suggest_structure.csv)")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    csv_path = args.csv or join(raw_dicom_folder, "suggest_structure.csv")
    if not os.path.isfile(csv_path):
        print("CSV not found at %s. Run nnDoseNet_0-2a_suggest_structure_csv first." % csv_path)
        return
    csv_to_structure_organ_json(csv_path, output_path=raw_dicom_folder)


def suggest_structure_entry():
    """nnDoseNet_0-2_suggest_structure: = 0-2a + 0-2b combined."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Match ROI names to canonical organs and generate structure_organ.json."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-organ_dict', type=str, default=None,
                        help="Path to organ_dictionary.json (default: bundled)")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    roi_json = join(raw_dicom_folder, args.trts + "_roi_list_summary.json")
    if not os.path.isfile(roi_json):
        print("ROI list not found at %s. Run nnDoseNet_0-1_list_roi first." % roi_json)
        return
    suggest_structure_from_roi_list(
        roi_json, organ_dictionary_path=args.organ_dict,
        output_path=raw_dicom_folder, output_csv_only=False,
    )


def list_target_entry():
    """nnDoseNet_0-3_list_target: List PTV/Target structures and compute prescriptions."""
    import argparse
    parser = argparse.ArgumentParser(
        description="List PTV/Target structures per patient, optionally compute D90/D95/D98 prescriptions."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('--prescription', type=str, default=None,
                        help="Prescription mode: blank (list only), a number (Gy), or 'D90'/'D95'/'D98' (compute from dose)")
    parser.add_argument('-out', type=str, default=None,
                        help="Output path for structure_target.json")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    prescription = args.prescription
    if prescription:
        try:
            prescription = float(prescription)
        except ValueError:
            pass  # keep as string (D90/D95/D98)
    write_target_summary_json(
        raw_dicom_folder, tr_ts=args.trts,
        universal_prescription=prescription,
        output_path=args.out,
    )


# =============================================================================
# Stage 1 — DICOM Conversion, Labeling & Beam Export
# =============================================================================

def convert_dicom_image_entry():
    """nnDoseNet_1-1_convert_dicom_image: Convert DICOM to NIfTI (CT + dose + structures)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert DICOM files to NIfTI: CT images, dose maps, and structures."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=3,
                        help="Number of processes")
    parser.add_argument('-structure', type=str, default=None,
                        help="Path to structure.json for structure conversion")
    parser.add_argument('-target_summary', type=str, default=None,
                        help="Optional patient-specific target-summary JSON; creates combined_target.nii.gz with prescription Gy x 100")
    parser.add_argument('--organized_dicom', default=False, action="store_true",
                        help="Skip DICOM organization")
    parser.add_argument('--continue', dest='continue_flag', default=False, action="store_true",
                        help="Skip patients that already have Nifti files")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    if not args.organized_dicom:
        organize_dicom(join(raw_dicom_folder, args.trts), num_processes=args.nps)
    format_dataset_from_dicom(raw_dicom_folder, args.trts, args.nps, args.structure, args.target_summary)


def label_nifti_entry():
    """nnDoseNet_1-2_label_nifti: Label and combine NIfTI, copy to raw folder."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Label NIfTI structures with integer values, combine, and copy to nnDose_raw."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-id', type=str, required=True,
                        help="[REQUIRED] Dataset ID (e.g. 012)")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=3,
                        help="Number of processes")
    parser.add_argument('-structure', type=str, default=None,
                        help="Path to structure.json for labeling")
    parser.add_argument('-target_summary', type=str, default=None,
                        help="Optional target-summary JSON; use combined_target.nii.gz for the Target channel")
    parser.add_argument('--continue', dest='continue_flag', default=False, action="store_true",
                        help="Skip patients that already have labeled files")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    organize_nifti(join(raw_dicom_folder, args.trts))
    format_dataset_from_nifti(
        join(raw_dicom_folder, args.trts), args.id,
        dataset_name_only=args.name, tr_ts=args.trts,
        num_processes=args.nps, structure_label=args.structure,
        target_summary_path=args.target_summary,
    )


def beam_export_entry():
    """nnDoseNet_1-3_beam_export: Generate beam maps from DICOM plan data."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate beam maps from DICOM plan data. Default: BEV mode."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=1,
                        help="Number of processes (BEV mode only)")
    parser.add_argument('-structure', type=str, default=None,
                        help="Optional structure JSON; use the same file supplied to conversion and labeling")
    parser.add_argument('--template', default=False, action="store_true",
                        help="Use template-based beam (shift/rotate pre-generated templates)")
    parser.add_argument('--gpu', default=False, action="store_true",
                        help="Use GPU (PyTorch CUDA) for BEV beam calculations")
    parser.add_argument('--use_field_only', default=False, action="store_true",
                        help="Use field-only mode (no divergence) for BEV")
    parser.add_argument('--continue', dest='continue_flag', default=False, action="store_true",
                        help="Skip patients that already have beam files")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    beam_export(
        raw_dicom_folder, tr_ts=args.trts,
        num_processes=args.nps,
        use_template=args.template,
        use_gpu=args.gpu,
        use_field_only=args.use_field_only,
        structure_json=args.structure,
    )


def format_dataset_entry():
    """nnDoseNet_1_format_dataset: = _1-1_ + _1-2_ + _1-3_ combined."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Complete Stage 1: Convert DICOM to NIfTI, label structures, generate beams, copy to raw folder."
    )
    parser.add_argument('-name', type=str, required=True,
                        help="[REQUIRED] Dataset name in nnDose_dicom")
    parser.add_argument('-id', type=str, required=True,
                        help="[REQUIRED] Dataset ID (e.g. 012)")
    parser.add_argument('-trts', type=str, default='Tr',
                        help="'Tr' for Training, 'Ts' for Testing")
    parser.add_argument('-nps', type=int, default=3,
                        help="Number of processes")
    parser.add_argument('-structure', type=str, default=None,
                        help="Path to structure.json for conversion and labeling")
    parser.add_argument('-target_summary', type=str, default=None,
                        help="Optional patient-specific target-summary JSON; creates and uses combined_target.nii.gz")
    parser.add_argument('--organized_dicom', default=False, action="store_true",
                        help="Skip DICOM organization")
    parser.add_argument('--template', default=False, action="store_true",
                        help="Use template-based beam generation")
    parser.add_argument('--gpu', default=False, action="store_true",
                        help="Use GPU for BEV beam calculations")
    parser.add_argument('--continue', dest='continue_flag', default=False, action="store_true",
                        help="Skip patients that already have output files")
    args, _ = parser.parse_known_args()
    raw_dicom_folder = join(nnDose_dicom, args.name)
    if not args.organized_dicom:
        organize_dicom(join(raw_dicom_folder, args.trts), num_processes=args.nps)
    format_dataset_from_dicom(raw_dicom_folder, args.trts, args.nps, args.structure, args.target_summary)
    organize_nifti(join(raw_dicom_folder, args.trts))
    format_dataset_from_nifti(
        join(raw_dicom_folder, args.trts), args.id,
        dataset_name_only=args.name, tr_ts=args.trts,
        num_processes=args.nps, structure_label=args.structure,
        target_summary_path=args.target_summary,
    )
    beam_export(
        raw_dicom_folder, tr_ts=args.trts,
        num_processes=args.nps,
        use_template=args.template,
        use_gpu=args.gpu,
        structure_json=args.structure,
    )


# Legacy aliases for backward compatibility
organize_and_convert_dicom_entry = convert_dicom_image_entry
label_nifti_and_final_formatting_entry = label_nifti_entry


if __name__ == '__main__':
    format_dataset_entry()
