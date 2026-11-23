from nndosenet.dicom_bridge.dicom2nifti import DicomConverter, NiftiLabeler
import numpy as np
import SimpleITK as sitk
import os
import re
import multiprocessing
import glob
import json
from tqdm import tqdm
from time import sleep
import shutil
import pandas as pd
from nndosenet.paths import nnDose_raw, nnDose_prediction
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile
import pydicom


def sitk_read_dcm(in_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    dicom = reader.Execute()
    return dicom
def pydicom_get_modality(in_path):
    try:
        ds = pydicom.dcmread(in_path)
        return ds.Modality

    except:
        print("cannot read file ",in_path, "with pydicom. please check if it;s dicom file.")
        return "Extras"

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

#TODO: change to find basefolder method
def create_folderlist(path,folder_list=[],overwrite=False):
    for sub_folder in folder_list:
        if not os.path.isdir(join(path,sub_folder)):
            os.makedirs(join(path,sub_folder))
        elif overwrite:
            shutil.rmtree(join(path,sub_folder))
            os.makedirs(join(path,sub_folder))

def sanity_check_match_roi():
    #TODO: load match roi summary if none check if structure set is all good
    pass
def sanity_check_dcm_find():
    pass
def organize_dicom_patient(DATA_FOLDER, patient_folder):
    """
    Helper function to organize dicom files for a single patient
    """
    if not os.path.isdir(join(DATA_FOLDER, patient_folder)):
        return {patient_folder: False}

    patient_path = join(DATA_FOLDER, patient_folder)
    create_folderlist(patient_path, folder_list=["Nifti","Dicom"])

    type_list = {"CT":"CT", "RTDOSE":"RD", "RTSTRUCT":"RS", "RTPLAN":"RP","Extras":"Extras"}
    create_folderlist(join(patient_path,"Dicom"), folder_list = list(type_list.values()))

    # Find all .dcm files recursively in the patient folder
    dcm_files = glob.glob(join(DATA_FOLDER, patient_folder, "**/*.dcm"), recursive=True)

    for dcm_path in dcm_files:
        if os.path.isfile(dcm_path):
            modality = pydicom_get_modality(dcm_path)
            # Get just the filename from the full path
            filename = os.path.basename(dcm_path)
            # Move the file to the appropriate modality folder
            shutil.move(dcm_path, join(DATA_FOLDER, patient_path, "Dicom", type_list[modality], filename))

    return {patient_folder: True}

def organize_dicom(DATA_FOLDER, num_processes=3):
    """
    DATA_FOLDER=nnDose_dicom/Dataset/Tr
    CT for CT image, RS for structure of image, RD or dose map (just 1 RD file please)
    The data structure of DATA_FOLDER should be like this
    -DATA_FOLDER
        -patient 1
            -CT.12345678.91011112.dcm
            -CT.12345678.91011113.dcm
            -CT.12345678.91011114.dcm
            ...
            -RS.12345678.91011117.dcm
            -RS.12345678.91011118.dcm
            ...
            -RD.12345678.91011119.dcm
        -patient 2
            -CT.22345678.91011112.dcm
            -CT.22345678.91011113.dcm
            -CT.22345678.91011114.dcm
            ...
            -RS.22345678.91011117.dcm
            -RS.22345678.91011118.dcm
            ...
            -RD.22345678.91011119.dcm
        -patient 3
        ...

    After apply organize_dicom DATA_FOLDER should be like this
    -DATA_FOLDER
        -0001
            -Nifti
            -Dicom
                -CT
                    -CT.12345678.91011112.dcm
                    -CT.12345678.91011113.dcm
                    -CT.12345678.91011114.dcm
                    ...
                -RS
                    -RS.12345678.91011117.dcm
                    -RS.12345678.91011118.dcm
                    ...
                -RD
                    -RD.12345678.91011119.dcm
                -extra
                    ....

        -0002
            -Nifti
            -Dicom
                -CT
                    -CT.22345678.91011112.dcm
                    -CT.22345678.91011113.dcm
                    -CT.22345678.91011114.dcm
                    ...
                -RS
                    -RS.22345678.91011117.dcm
                    -RS.22345678.91011118.dcm
                    ...
                -RD
                    -RD.22345678.91011119.dcm
                -extra
                    ....
        -0003
        ...
    """
    patients = sorted(os.listdir(DATA_FOLDER))
    r = []
    results_dict = {}

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for patient in patients:
            r.append(p.starmap_async(organize_dicom_patient,
                                     ((DATA_FOLDER, patient),)))

        remaining = list(range(len(patients)))
        workers = [j for j in p._pool]
        with tqdm(desc="Organizing DICOM files", total=len(patients)) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                done = [i for i in remaining if r[i].ready()]
                for i in done:
                    pbar.update()
                    results_dict.update(r[i].get()[0])
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)

    print('Finished organizing DICOM files in directory: ' + DATA_FOLDER)

def organize_nifti(DATA_FOLDER):
    """
    DATA_FOLDER=nnUNet_dicom/Dataset/Tr
    After apply organize_nifti DATA_FOLDER should be like this
    -DATA_FOLDER
        -0001
            -Nifti
                -ct.nii.gz
                -dose.nii.gz
                -structure1.nii.gz
                -structure2.nii.gz
                -structure3.nii.gz
            -Dicom

        -0002
            -Nifti
                -ct.nii.gz
                -dose.nii.gz
                -structure1.nii.gz
                -structure2.nii.gz
                -structure3.nii.gz
            -Dicom
        -0003
        ...
    """
    #print(sorted(os.listdir(DATA_FOLDER)))
    for patient_folder in sorted(os.listdir(DATA_FOLDER)):
        if not os.path.isdir(join(DATA_FOLDER,patient_folder)):
            continue
        patient_path=join(DATA_FOLDER,patient_folder)
        create_folderlist(patient_path,folder_list=["Nifti"])
        niftis=glob.glob(join(patient_path,"*.nii.gz"))
        for nifti in niftis:
            shutil.move(nifti,join(patient_path,"Nifti",))

    print(' finish organizing the the data in dir: '+DATA_FOLDER)
def dicom_convert(folder,patient,structure_json=None,target_summary=None):
    try:
        target_prescriptions = None
        if target_summary is not None:
            if patient not in target_summary:
                raise KeyError(f"Patient {patient} is missing from the target summary")
            target_prescriptions = target_summary[patient]
        temp=DicomConverter(
            folder,
            patient_folder=patient,
            structure_name=structure_json,
            target_prescriptions=target_prescriptions,
        ).run()
    except Exception as e:
        print(f"WARNING: Failed to convert {patient}: {e}")
        temp={"error": str(e)}
    return {patient:temp}

def format_dataset_from_dicom(folder,tr_ts,num_processes=3,structure_json=None,target_summary_path=None):
    target_summary = None
    if target_summary_path is not None:
        if not isfile(target_summary_path):
            raise FileNotFoundError(f"Target summary does not exist: {target_summary_path}")
        target_summary = load_json(target_summary_path)
        if not isinstance(target_summary, dict):
            raise ValueError(f"Target summary must contain a JSON object: {target_summary_path}")

    folder=join(folder,tr_ts)

    patients=sorted([p for p in os.listdir(folder) if os.path.isdir(join(folder, p))])

    results_dict = {}

    if num_processes <= 1:
        # Single-process mode (no spawn issues, clearer stack traces)
        for patient in tqdm(patients, desc="converting Dicom2Nifti"):
            result = dicom_convert(folder, patient, structure_json, target_summary)
            results_dict.update(result)
    else:
        r=[]
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for patient in patients:
                r.append(p.starmap_async(dicom_convert,
                                         ((folder,patient,structure_json,target_summary),)))

            remaining = list(range(len(patients)))
            workers = [j for j in p._pool]
            with tqdm(desc="converting Dicom2Nifti", total=len(patients) )as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                    done = [i for i in remaining if r[i].ready()]
                    for i in done:
                        pbar.update()
                        results_dict.update(r[i].get()[0])
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    if target_summary is not None:
        failed_patients = [patient for patient, result in results_dict.items() if "error" in result]
        if failed_patients:
            raise RuntimeError(
                "DICOM conversion failed for target-summary cases: " + ", ".join(failed_patients)
            )

    # for i in r:
    #     results_dict.update(i.get()[0]) #todo test it in while done , update there

    #print(results_dict)
    save_sum_name = os.path.join(os.path.dirname(folder),tr_ts+'_roi_summary.csv')#todo:  to base +trts
    results_dict = dict(sorted(results_dict.items()))
    dataFrame = pd.DataFrame(results_dict).T#.sort_index()

    # dataFrame=dataFrame.rename(columns={dataFrame.columns[0]:"patients"})
    # dataFrame.sort_values(by=['patients'])
    dataFrame.to_csv(save_sum_name)
    save_json(results_dict,save_sum_name.replace('.csv', '.json'))

def nifti_label(folder,patient_folder,patient_id,dataset_name,dataset_id,tr_ts,structure_label=None,target_summary_path=None):
    NiftiLabeler(
        folder,
        patient_folder,
        patient_id,
        dataset_name,
        dataset_id,
        tr_ts,
        structure_label=structure_label,
        target_summary_path=target_summary_path,
    ).run()

def format_dataset_from_nifti(folder,dataset_id,dataset_name_only=None,tr_ts="Tr",num_processes=8,structure_label=None,
                              target_summary_path=None):
    if target_summary_path is not None and not isfile(target_summary_path):
        raise FileNotFoundError(f"Target summary does not exist: {target_summary_path}")
    if dataset_name_only == None:
        dataset_name_only = folder.split("/")[-2]

    # Incoming DICOM folders may contain accidental leading/trailing whitespace.
    # Retain the exact source folder for reading, but never propagate that
    # whitespace into DatasetNNN_* output names or prediction directories.
    dataset_name_only = str(dataset_name_only).strip()
    if not dataset_name_only:
        raise ValueError("dataset_name_only must contain a non-whitespace dataset name")

    dataset_name="Dataset"+dataset_id+"_"+dataset_name_only

    patients = sorted([p for p in os.listdir(folder) if os.path.isdir(join(folder, p))])
    # Use stable, anonymized IDs based on sorted case-folder names. DICOM PatientID
    # must stay in the source DICOM only; patient{Tr,Ts}.csv records the mapping.
    id_list = list(range(1, len(patients) + 1))


    if tr_ts=="Tr":
        save_folder=join(nnDose_raw,dataset_name)
    else:
        save_folder=join(nnDose_prediction,dataset_name_only,dataset_name,"GroundTruth")
    if not os.path.isdir(join(save_folder)):
        os.makedirs(join(save_folder))
        print("Create folder in : ",join(save_folder))

    if not os.path.isdir(join(save_folder,"images"+tr_ts)):
        os.makedirs(join(save_folder,"images"+tr_ts))
    if not os.path.isdir(join(save_folder,"labels"+tr_ts)):
        os.makedirs(join(save_folder,"labels"+tr_ts))

    print("Labeling the contours")
    if num_processes <= 1:
        # Single-process mode
        for patient, id_val in tqdm(zip(patients, id_list), desc="labeling contours", total=len(patients)):
            nifti_label(folder, patient, id_val, dataset_name, dataset_id, tr_ts, structure_label, target_summary_path)
    else:
        r=[]
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for patient,id in zip(patients,id_list):
                r.append(p.starmap_async(nifti_label,
                                         ((folder,patient,id,dataset_name,dataset_id,tr_ts,structure_label,target_summary_path),)))
            remaining = list(range(len(patients)))
            workers = [j for j in p._pool]
            with tqdm(desc="labeling contours", total=len(patients) )as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck.')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    df = pd.DataFrame({'id': id_list,
                   'patient': patients})
    df.to_csv(join(save_folder,'patient'+tr_ts+'.csv'), index=False)

    if tr_ts=="Tr":
        write_dataset_json(patient_num=len(patients),dataset_name=dataset_name)
    if structure_label != None:
        shutil.copy2(join(structure_label),join(save_folder,"structure.json"))
    else:
        shutil.copy2(join(os.path.dirname(folder),'structure.json'),join(save_folder,"structure.json"))


    print("Finish ! All needed file is in ",join(save_folder),"now.")

def write_dataset_json(patient_num,dataset_name):
    dict={"labels": {"background": "0", "dose": ""},
          "channel_names": {"0": "CT", "1": "Target", "2": "OAR", "3": "BODY"},
          "numTraining": patient_num,
          "file_ending": ".nii.gz",
          "mask_channel":[1,2,3]}
    json_object = json.dumps(dict, indent=4)

    with open(join(nnDose_raw,dataset_name,"dataset.json"), "w") as outfile:
        outfile.write(json_object)


# =============================================================================
# Stage 0 — DICOM Organization & Structure Assistance
# =============================================================================

# Default ROI types and name patterns to exclude (setup points, markers, isocenter, etc.)
DEFAULT_EXCLUDE_ROI_TYPES = frozenset({
    "MARKER", "ISOCENTER", "REGISTRATION", "FIXATION", "PIXELVALUE", "SUPPORT",
})
DEFAULT_EXCLUDE_ROI_NAME_SUBSTRINGS = (
    "marker", "isocenter", "setup", " point", "_point", "registration", "fixation",
)
# ROI names starting with "_" are always excluded (e.g. _Setup, _Isocenter)
EXCLUDE_NAMES_STARTING_WITH_UNDERSCORE = True

# Default organ dictionary path (canonical name -> synonyms)
_DEFAULT_ORGAN_DICTIONARY_PATH = join(os.path.dirname(__file__), "organ_dictionary.json")


def _list_roi_names_for_patient(
    patient_path,
    exclude_types=None,
    exclude_name_substrings=None,
    exclude_names_starting_with_underscore=None,
):
    """
    Read RT Struct (RS) DICOM in patient_path/Dicom/RS and return ROI names,
    excluding setup points/markers by type and name patterns.
    Does not load CT or convert to Nifti.
    Returns:
        dict with keys: patient_folder, patient_id, roi_names (list), excluded_names (list),
                        roi_count, excluded_count, roi_details (list of {roi_name, roi_type})
    roi_names and excluded_names are disjoint; excluded ROIs never appear in roi_names.
    """
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_ROI_TYPES
    if exclude_name_substrings is None:
        exclude_name_substrings = DEFAULT_EXCLUDE_ROI_NAME_SUBSTRINGS
    if exclude_names_starting_with_underscore is None:
        exclude_names_starting_with_underscore = EXCLUDE_NAMES_STARTING_WITH_UNDERSCORE
    patient_folder = os.path.basename(patient_path.rstrip(os.sep))
    rs_folder = join(patient_path, "Dicom", "RS")
    result = {
        "patient_folder": patient_folder,
        "patient_id": "",
        "roi_names": [],
        "excluded_names": [],
        "roi_count": 0,
        "excluded_count": 0,
        "roi_details": [],
    }
    if not os.path.isdir(rs_folder):
        return result
    all_roi_names = []
    all_roi_details = []
    patient_id_set = False
    for rs_name in sorted(os.listdir(rs_folder)):
        rs_file = join(rs_folder, rs_name)
        if not os.path.isfile(rs_file) or not (rs_name.lower().endswith(".dcm") or rs_name.lower().endswith(".dicom")):
            continue
        try:
            ds = pydicom.dcmread(rs_file)
            if not patient_id_set:
                result["patient_id"] = getattr(ds, "PatientID", "") or ""
                patient_id_set = True
            seq = getattr(ds, "StructureSetROISequence", None)
            if seq is None:
                continue
            for item in seq:
                roi_name = getattr(item, "ROIName", None)
                if roi_name is None:
                    continue
                roi_name = str(roi_name).strip()
                if not roi_name or roi_name in all_roi_names:
                    continue
                roi_type = getattr(item, "ROIType", None)
                if roi_type is not None:
                    roi_type = str(roi_type).strip().upper()
                all_roi_names.append(roi_name)
                all_roi_details.append({"roi_name": roi_name, "roi_type": roi_type or ""})
        except Exception:
            continue
    included = []
    excluded = []
    for detail in all_roi_details:
        name = detail["roi_name"]
        rtype = detail["roi_type"]
        if exclude_names_starting_with_underscore and name.startswith("_"):
            excluded.append(name)
            continue
        if rtype and rtype in exclude_types:
            excluded.append(name)
            continue
        if any(s in name.lower() for s in exclude_name_substrings):
            excluded.append(name)
            continue
        included.append(name)
    excluded_set = set(excluded)
    result["roi_names"] = [n for n in included if n not in excluded_set]
    result["excluded_names"] = excluded
    result["roi_count"] = len(included)
    result["excluded_count"] = len(excluded)
    result["roi_details"] = all_roi_details
    return result


def list_roi_names_from_dicom(
    folder,
    tr_ts="Tr",
    output_dir=None,
    exclude_types=None,
    exclude_name_substrings=None,
    num_processes=1,
):
    """
    List ROI structure names from RT Struct DICOM for all patients (after organize_dicom,
    before DICOM->Nifti conversion). Excludes setup points/markers by type and name.
    Writes a CSV and JSON summary.

    Args:
        folder: Base folder containing tr_ts subfolder (e.g. nnDose_dicom/openkbp).
        tr_ts: 'Tr' or 'Ts'.
        output_dir: Where to write roi_list_summary.csv and .json. Default: same as folder.
        num_processes: Parallel workers (default 1).

    Returns:
        results_dict: { patient_folder: { patient_id, roi_names, excluded_names, ... } }
    """
    data_folder = join(folder, tr_ts)
    if output_dir is None:
        output_dir = os.path.dirname(data_folder)
    patients = sorted([p for p in os.listdir(data_folder) if os.path.isdir(join(data_folder, p))])
    if not patients:
        return {}
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_ROI_TYPES
    if exclude_name_substrings is None:
        exclude_name_substrings = DEFAULT_EXCLUDE_ROI_NAME_SUBSTRINGS

    def _task(patient_folder):
        patient_path = join(data_folder, patient_folder)
        return patient_folder, _list_roi_names_for_patient(
            patient_path, exclude_types=exclude_types, exclude_name_substrings=exclude_name_substrings
        )

    if num_processes <= 1:
        results_list = [_task(p) for p in tqdm(patients, desc="Listing ROI names")]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(num_processes) as pool:
            results_list = list(tqdm(
                pool.starmap(_task, [(p,) for p in patients]),
                total=len(patients),
                desc="Listing ROI names",
            ))
    results_dict = {}
    for patient_folder, row in results_list:
        results_dict[patient_folder] = {
            "patient_id": row["patient_id"],
            "roi_names": row["roi_names"],
            "excluded_names": row["excluded_names"],
            "roi_count": row["roi_count"],
            "excluded_count": row["excluded_count"],
            "roi_details": row["roi_details"],
        }
    results_dict = dict(sorted(results_dict.items()))
    # CSV
    csv_path = join(output_dir, tr_ts + "_roi_list_summary.csv")
    rows = []
    for pf, data in results_dict.items():
        rows.append({
            "patient_folder": pf,
            "patient_id": data["patient_id"],
            "roi_count": data["roi_count"],
            "excluded_count": data["excluded_count"],
            "roi_names": json.dumps(data["roi_names"]) if data["roi_names"] else "[]",
            "excluded_names": json.dumps(data["excluded_names"]) if data["excluded_names"] else "[]",
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # JSON
    json_path = join(output_dir, tr_ts + "_roi_list_summary.json")
    save_json(results_dict, json_path)
    print("ROI list summary written to %s and %s" % (csv_path, json_path))
    return results_dict



def _canonical_specificity_order(canonical):
    """Sort key: process specific (e.g. Kidney_L, Kidney_R) before generic (Kidney) so matching prefers specific."""
    c = canonical
    has_lateral = ("_L" in c or "_R" in c or "Left" in c or "Right" in c or c.endswith("_L") or c.endswith("_R"))
    return (0 if has_lateral else 1, -len(c), c)


def _normalize_roi_for_match(name):
    """Normalize ROI name for matching: lowercase, single underscores, no leading/trailing space."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Pretrain-compatible OAR labels 1..18 (fixed for transfer learning)
PRETRAIN_OAR_LABELS = {
    "1": ["Body"], "2": ["Spine"], "3": ["Lung_L", "Lung_R"],
    "4": ["Esophagus"], "5": ["Heart"], "6": ["Liver"],
    "7": ["Bladder"], "8": ["Rectum"], "9": ["Femur_Head_L", "Femur_Head_R"],
    "10": ["Bowel"], "11": ["Glnd_Thyroid"], "12": ["Trachea"],
    "13": ["Carina"], "14": ["Bronchus_Main_R", "Bronchus_Main_L"],
    "15": ["Mandible"], "16": ["BrainStem"], "17": ["Larynx"],
    "18": ["Parotid_L", "Parotid_R"],
}


def suggest_structure_from_roi_list(
    roi_list_summary_path_or_dict,
    organ_dictionary_path=None,
    output_path=None,
    include_unmatched_as_other=False,
    output_csv_only=False,
):
    """
    Categorize ROI names from roi_list_summary using an organ dictionary and generate
    a structure_organ.json and editable CSV. OAR labels 1..18 match the pretrain
    template for transfer learning.

    Args:
        roi_list_summary_path_or_dict: Path to Tr_roi_list_summary.json or a dict.
        organ_dictionary_path: Path to organ_dictionary.json. Default: bundled.
        output_path: Directory to write outputs. Default: same dir as ROI list.
        include_unmatched_as_other: If True, add 'Other' category with unmatched ROIs.
        output_csv_only: If True, only output suggest_structure.csv (for 0-2a).

    Returns:
        dict with keys: structure_name, structure_label, unmatched, report.
    """
    if organ_dictionary_path is None:
        organ_dictionary_path = _DEFAULT_ORGAN_DICTIONARY_PATH
    if not os.path.isfile(organ_dictionary_path):
        raise FileNotFoundError("Organ dictionary not found: %s" % organ_dictionary_path)
    organ_dict = load_json(organ_dictionary_path)

    # Load ROI list
    if isinstance(roi_list_summary_path_or_dict, dict):
        roi_data = roi_list_summary_path_or_dict
        base_dir = None
    else:
        roi_data = load_json(roi_list_summary_path_or_dict)
        base_dir = os.path.dirname(roi_list_summary_path_or_dict)
    if output_path is None and base_dir:
        output_path = base_dir

    # Collect all unique ROI names
    all_roi_names_ordered = []
    seen = set()
    for pf, data in sorted(roi_data.items()):
        for name in data.get("roi_names", []):
            if name and name not in seen:
                seen.add(name)
                all_roi_names_ordered.append(name)

    # Build synonym -> canonical map (normalized), specific-first
    canonicals_ordered = sorted(
        [c for c in organ_dict.keys() if not c.startswith("_") and isinstance(organ_dict.get(c), list)],
        key=_canonical_specificity_order,
    )
    norm_to_canonical = {}
    for canonical in canonicals_ordered:
        synonyms = organ_dict[canonical]
        for syn in synonyms:
            n = _normalize_roi_for_match(syn)
            if n and n not in norm_to_canonical:
                norm_to_canonical[n] = canonical

    # Map each ROI name to a canonical
    canonical_to_actual = {}
    per_patient_canonical = {}
    unmatched = []
    for roi_name in all_roi_names_ordered:
        norm = _normalize_roi_for_match(roi_name)
        canonical = norm_to_canonical.get(norm)
        if canonical is None:
            unmatched.append(roi_name)
            continue
        if canonical not in canonical_to_actual:
            canonical_to_actual[canonical] = []
        if roi_name not in canonical_to_actual[canonical]:
            canonical_to_actual[canonical].append(roi_name)

    # Per-patient canonical mapping
    per_patient_unmatched = {}
    for pf, data in sorted(roi_data.items()):
        per_patient_canonical[pf] = {}
        not_used = []
        for roi_name in data.get("roi_names", []):
            if not roi_name:
                continue
            norm = _normalize_roi_for_match(roi_name)
            canonical = norm_to_canonical.get(norm)
            if canonical is None:
                not_used.append(roi_name)
                continue
            if canonical not in per_patient_canonical[pf]:
                per_patient_canonical[pf][canonical] = []
            if roi_name not in per_patient_canonical[pf][canonical]:
                per_patient_canonical[pf][canonical].append(roi_name)
        per_patient_unmatched[pf] = not_used

    # Build structure_name using organ_dict canonical names + actual ROI names from data
    structure_name = {}
    for c in sorted(canonical_to_actual.keys()):
        structure_name[c] = list(canonical_to_actual[c])
    if include_unmatched_as_other and unmatched:
        structure_name["Other"] = list(unmatched)

    # Build structure_label: pretrain OAR 1..18 + extra for additional found organs
    structure_label = {"OAR": {}}
    for label_key in sorted(PRETRAIN_OAR_LABELS.keys(), key=int):
        canonicals_for_label = PRETRAIN_OAR_LABELS[label_key]
        matched = [c for c in canonicals_for_label if c in structure_name]
        if matched:
            structure_label["OAR"][label_key] = matched
    # Extra organs get labels 19, 20, ...
    pretrain_set = set()
    for v in PRETRAIN_OAR_LABELS.values():
        pretrain_set.update(v)
    next_label = 19
    extra_canonicals = sorted([c for c in structure_name.keys() if c not in pretrain_set and c != "Other"])
    # Group L/R pairs
    bases_with_pairs = set()
    for c in extra_canonicals:
        if c.endswith("_L"):
            bases_with_pairs.add(c[:-2])
        if c.endswith("_R"):
            bases_with_pairs.add(c[:-2])
    used_in_group = set()
    for base in sorted(bases_with_pairs):
        group = []
        for suffix in ("_L", "_R"):
            cand = base + suffix
            if cand in structure_name and cand in extra_canonicals:
                group.append(cand)
        for generic in (base, base + "s"):
            if generic in structure_name and generic in extra_canonicals and generic not in group:
                group.append(generic)
        if group:
            group.sort()
            structure_label["OAR"][str(next_label)] = group
            next_label += 1
            used_in_group.update(group)
    for c in sorted(extra_canonicals):
        if c in used_in_group:
            continue
        structure_label["OAR"][str(next_label)] = [c]
        next_label += 1

    # Build report
    report_lines = [
        "Suggested structure_name from organ dictionary and ROI list.",
        "OAR 1..18: pretrain-compatible (same as trained model). OAR 19+: extra ROIs.",
        "Matched categories: %d" % len(structure_name),
        "Unmatched ROI names (%d): %s" % (len(unmatched), unmatched),
        "",
        "Per category (canonical -> actual names in dataset):",
    ]
    for c in sorted(structure_name.keys()):
        report_lines.append("  %s -> %s" % (c, structure_name[c]))

    result = {
        "structure_name": structure_name,
        "structure_label": structure_label,
        "unmatched": unmatched,
        "report": "\n".join(report_lines),
    }

    if output_path:
        # CSV for user editing (0-2a output)
        out_csv = join(output_path, "suggest_structure.csv")
        def _label_for(c):
            for k, v in structure_label["OAR"].items():
                if c in v:
                    return "OAR", k
            return "OAR", ""
        csv_rows = []
        for c in sorted(structure_name.keys()):
            names = structure_name[c]
            label_type, label_val = _label_for(c)
            # Count how many patients have this structure
            patient_list = [pf for pf in per_patient_canonical if c in per_patient_canonical[pf]]
            csv_rows.append({
                "Canonical_Name": c,
                "ROI_Names_Found": json.dumps(names) if names else "[]",
                "Assigned_Label": label_val,
                "Patient_Count": len(patient_list),
                "Patients": json.dumps(patient_list[:10]),  # first 10 for readability
            })
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

        if not output_csv_only:
            # Write structure_organ.json (ready for pipeline)
            out_dict = {"structure_name": structure_name, "structure_label": structure_label}
            out_json = join(output_path, "structure_organ.json")
            save_json(out_dict, out_json)
            # Also write legacy structure.json for backward compat
            save_json(out_dict, join(output_path, "structure.json"))
            # Report
            out_report = join(output_path, "structure_suggested_report.txt")
            with open(out_report, "w") as f:
                f.write(result["report"])
            print("Structure suggestion written to %s, %s, %s" % (out_csv, out_json, out_report))
        else:
            print("Structure suggestion CSV written to %s (edit and run 0-2b to generate JSON)" % out_csv)

    return result


def csv_to_structure_organ_json(csv_path, output_path=None):
    """
    Convert user-edited suggest_structure.csv to structure_organ.json.
    The CSV has columns: Canonical_Name, ROI_Names_Found, Assigned_Label, ...
    User should edit Assigned_Label (integer) and can correct Canonical_Name.

    Ensures Body is always label 1 and consistent labels across patients.
    """
    df = pd.read_csv(csv_path)
    if output_path is None:
        output_path = os.path.dirname(csv_path)

    structure_name = {}
    structure_label = {"OAR": {}}

    for _, row in df.iterrows():
        canonical = str(row.get("Canonical_Name", "")).strip()
        if not canonical:
            continue
        # Parse ROI names
        roi_names_raw = row.get("ROI_Names_Found", "[]")
        try:
            roi_names = json.loads(roi_names_raw) if isinstance(roi_names_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            roi_names = [roi_names_raw] if roi_names_raw else []

        label_val = str(row.get("Assigned_Label", "")).strip()
        if not label_val:
            continue

        structure_name[canonical] = roi_names if roi_names else [canonical]

        if label_val not in structure_label["OAR"]:
            structure_label["OAR"][label_val] = []
        structure_label["OAR"][label_val].append(canonical)

    # Ensure Body is label 1
    if "Body" in structure_name:
        # Remove Body from any other label
        for k, v in list(structure_label["OAR"].items()):
            if "Body" in v and k != "1":
                v.remove("Body")
                if not v:
                    del structure_label["OAR"][k]
        if "1" not in structure_label["OAR"]:
            structure_label["OAR"]["1"] = []
        if "Body" not in structure_label["OAR"]["1"]:
            structure_label["OAR"]["1"].insert(0, "Body")

    out_dict = {"structure_name": structure_name, "structure_label": structure_label}
    out_json = join(output_path, "structure_organ.json")
    save_json(out_dict, out_json)
    # Also write legacy structure.json
    save_json(out_dict, join(output_path, "structure.json"))
    print("structure_organ.json written to %s" % out_json)
    return out_dict


def _get_ptv_roi_names_for_patient(patient_path):
    """
    Same exclude logic as list_roi (0-1): no leading _, not in exclude types/substrings.
    Then keep only ROI names that contain 'PTV' or 'Target' (case insensitive).
    Returns list of ROI names (may be empty).
    """
    row = _list_roi_names_for_patient(patient_path)
    included = row.get("roi_names", [])
    return [n for n in included if "ptv" in n.lower() or "target" in n.lower()]


def _compute_dvh_metric_from_dicom(patient_path, roi_name, metric):
    """
    Compute D95, D99, or D90 (Gy) for one ROI from DICOM dose + RS.
    Uses first RD and first RS; structure mask is built on CT grid then resampled to dose grid.
    Returns float (Gy) or None if unavailable.
    """
    pct = {"D95": 5.0, "D99": 1.0, "D90": 10.0}.get(metric)
    if pct is None:
        return None
    rd_dir = join(patient_path, "Dicom", "RD")
    rs_dir = join(patient_path, "Dicom", "RS")
    ct_dir = join(patient_path, "Dicom", "CT")
    if not os.path.isdir(rd_dir) or not os.path.isdir(rs_dir) or not os.path.isdir(ct_dir):
        return None
    rd_files = sorted([f for f in os.listdir(rd_dir) if f.lower().endswith((".dcm", ".dicom"))])
    rs_files = sorted([f for f in os.listdir(rs_dir) if f.lower().endswith((".dcm", ".dicom"))])
    if not rd_files or not rs_files:
        return None
    try:
        # Read CT
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(ct_dir)
        if not series_ids:
            return None
        dicom_names = reader.GetGDCMSeriesFileNames(ct_dir, series_ids[0])
        reader.SetFileNames(dicom_names)
        ct_img = reader.Execute()
        # Read dose
        dose_ds = pydicom.dcmread(join(rd_dir, rd_files[0]))
        dose_arr = dose_ds.pixel_array.astype(np.float64) * float(dose_ds.DoseGridScaling)
        dose_img = sitk.GetImageFromArray(dose_arr)
        dose_img.SetOrigin(list(map(float, dose_ds.ImagePositionPatient)))
        gfo = getattr(dose_ds, "GridFrameOffsetVector", None)
        if gfo is not None and len(gfo) > 1:
            z_spacing = float(gfo[1]) - float(gfo[0])
        else:
            z_spacing = 1.0
        dose_img.SetSpacing([float(dose_ds.PixelSpacing[0]), float(dose_ds.PixelSpacing[1]), z_spacing])
        iop = getattr(dose_ds, "ImageOrientationPatient", None)
        if iop:
            vec_x = np.array(iop[0:3], dtype=np.float64)
            vec_y = np.array(iop[3:6], dtype=np.float64)
            vec_z = np.cross(vec_x, vec_y)
            direction = np.array([
                vec_x[0], vec_y[0], vec_z[0],
                vec_x[1], vec_y[1], vec_z[1],
                vec_x[2], vec_y[2], vec_z[2]], dtype=np.float64)
            dose_img.SetDirection(direction.tolist())
        from rt_utils import RTStructBuilder
        rtstruct = RTStructBuilder.create_from(ct_dir, join(rs_dir, rs_files[0]))
        try:
            mask_ct = rtstruct.get_roi_mask_by_name(roi_name)
        except Exception:
            return None
        if mask_ct is None or not np.any(mask_ct):
            return None
        # rt_utils mask is (rows, cols, slices) = (y, x, z); SimpleITK expects (z, y, x)
        mask_arr_ct = np.transpose(mask_ct.astype(np.float64), (2, 0, 1))
        mask_img = sitk.GetImageFromArray(mask_arr_ct)
        mask_img.CopyInformation(ct_img)
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(dose_img)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
        mask_on_dose = resample.Execute(mask_img)
        mask_arr = sitk.GetArrayFromImage(mask_on_dose)
        if mask_arr.shape != dose_arr.shape:
            return None
        vals = dose_arr[mask_arr > 0.5].astype(np.float64)
        if vals.size == 0:
            return None
        return float(np.percentile(vals, pct))
    except Exception:
        return None


def write_target_summary_json(
    raw_dicom_base_folder,
    tr_ts="Tr",
    universal_prescription=None,
    output_path=None,
    num_processes=1,
):
    """
    Write structure_target.json: one entry per patient with PTV/Target-containing structures.
    Uses same exclude logic as list_roi (0-1): structures not starting with _, not in
    exclude types/substrings, and whose name contains 'PTV' or 'Target'.

    Prescription per target:
    - universal_prescription is None or "": save prescription as blank (empty string).
    - universal_prescription is a number: all targets get that value (Gy).
    - universal_prescription is "D95", "D99", or "D90": compute that metric from
      DICOM dose + RS.

    DISCLAIMER: We recommend using the actual clinical prescription values when available.
    D90/D95/D98 computed from the dose distribution are approximations.
    All nnDoseNet-related published papers use the actual clinical prescription.

    Format: { "patient_folder": { "PTV_1": 12.5, "PTV Body": "" }, ... }
    """
    data_folder = join(raw_dicom_base_folder, tr_ts)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    patients = sorted(
        [p for p in os.listdir(data_folder) if os.path.isdir(join(data_folder, p))]
    )
    # Print disclaimer for DVH-based prescription
    if universal_prescription and str(universal_prescription).strip().upper() in ("D95", "D99", "D90"):
        print("=" * 80)
        print("DISCLAIMER: Using %s from dose distribution as prescription approximation." % universal_prescription)
        print("We recommend using the actual clinical prescription values when available.")
        print("All nnDoseNet-related published papers use the actual clinical prescription.")
        print("=" * 80)

    summary = {}
    for pf in tqdm(patients, desc="Target summary"):
        patient_path = join(data_folder, pf)
        ptv_names = _get_ptv_roi_names_for_patient(patient_path)
        if not ptv_names:
            summary[pf] = {}
            continue
        row = {}
        if universal_prescription is None or universal_prescription == "":
            for n in ptv_names:
                row[n] = ""
        elif isinstance(universal_prescription, (int, float)):
            for n in ptv_names:
                row[n] = float(universal_prescription)
        elif str(universal_prescription).strip().upper() in ("D95", "D99", "D90"):
            metric = str(universal_prescription).strip().upper()
            for n in ptv_names:
                val = _compute_dvh_metric_from_dicom(patient_path, n, metric)
                row[n] = round(val, 4) if val is not None else ""
        else:
            try:
                rx = float(universal_prescription)
                for n in ptv_names:
                    row[n] = rx
            except (TypeError, ValueError):
                for n in ptv_names:
                    row[n] = ""
        summary[pf] = row

    if output_path is None:
        output_path = join(raw_dicom_base_folder, "structure_target.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Target summary written to %s" % output_path)
    return output_path


def list_dicom_index(folder, tr_ts="Tr", output_dir=None):
    """
    Create a plan-centric DICOM index spreadsheet. One row per plan.
    Includes CT/Dose grid metadata and all beam parameters needed for
    template-based beam generation.

    Output columns:
        PatientFolder, PatientID, PlanLabel, PlanUID,
        RefStructUID, RefCTSeriesUID, CTSpacing, CTOrigin, CTDirection, CTSize,
        RefDoseUIDs, DoseSpacing, DoseOrigin, DoseDirection, DoseSize,
        NumTreatmentBeams, BeamNames, Isocenters, GantryAngles,
        CouchAngles, CollimatorAngles, ArcTypes, TotalCPs
    """
    data_folder = join(folder, tr_ts)
    if output_dir is None:
        output_dir = os.path.dirname(data_folder)

    patients = sorted([p for p in os.listdir(data_folder) if os.path.isdir(join(data_folder, p))])
    rows = []

    for p in tqdm(patients, desc="Building DICOM Index"):
        p_path = join(data_folder, p, "Dicom")
        ct_dir = join(p_path, "CT")
        rp_dir = join(p_path, "RP")
        rs_dir = join(p_path, "RS")
        rd_dir = join(p_path, "RD")

        # CT Series metadata cache
        ct_series_meta = {}
        if os.path.isdir(ct_dir):
            try:
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(ct_dir)
                for sid in series_ids:
                    filenames = reader.GetGDCMSeriesFileNames(ct_dir, sid)
                    if not filenames:
                        continue
                    reader.SetFileNames(filenames)
                    ct_img = reader.Execute()
                    ct_series_meta[sid] = {
                        "spacing": list(ct_img.GetSpacing()),
                        "origin": list(ct_img.GetOrigin()),
                        "direction": list(ct_img.GetDirection()),
                        "size": list(ct_img.GetSize()),
                    }
            except Exception:
                pass

        # RS: SOP UID -> referenced CT SeriesUID
        rs_uid_to_ct_series = {}
        if os.path.isdir(rs_dir):
            for fname in os.listdir(rs_dir):
                if not fname.lower().endswith((".dcm", ".dicom")):
                    continue
                try:
                    rs_ds = pydicom.dcmread(join(rs_dir, fname), stop_before_pixels=True)
                    rs_sop = getattr(rs_ds, "SOPInstanceUID", "")
                    ct_series_uid = ""
                    if "ReferencedFrameOfReferenceSequence" in rs_ds:
                        for fref in rs_ds.ReferencedFrameOfReferenceSequence:
                            if "RTReferencedStudySequence" in fref:
                                for study in fref.RTReferencedStudySequence:
                                    if "RTReferencedSeriesSequence" in study:
                                        for ser in study.RTReferencedSeriesSequence:
                                            ct_series_uid = getattr(ser, "SeriesInstanceUID", "")
                                            break
                    rs_uid_to_ct_series[rs_sop] = ct_series_uid
                except Exception:
                    pass

        # RD: Plan UID -> dose info
        plan_to_dose_info = {}
        if os.path.isdir(rd_dir):
            for fname in sorted(os.listdir(rd_dir)):
                if not fname.lower().endswith((".dcm", ".dicom")):
                    continue
                try:
                    rd_ds = pydicom.dcmread(join(rd_dir, fname))
                    ref_plan_uid = ""
                    if "ReferencedRTPlanSequence" in rd_ds:
                        ref_plan_uid = rd_ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                    dose_sop = getattr(rd_ds, "SOPInstanceUID", "")
                    dose_arr = rd_ds.pixel_array
                    dose_spacing = [float(rd_ds.PixelSpacing[0]), float(rd_ds.PixelSpacing[1]), 0.0]
                    if hasattr(rd_ds, "GridFrameOffsetVector") and len(rd_ds.GridFrameOffsetVector) > 1:
                        dose_spacing[2] = float(rd_ds.GridFrameOffsetVector[1] - rd_ds.GridFrameOffsetVector[0])
                    dose_origin = [float(v) for v in rd_ds.ImagePositionPatient]
                    dose_size = [int(dose_arr.shape[2]), int(dose_arr.shape[1]), int(dose_arr.shape[0])]
                    iop = getattr(rd_ds, "ImageOrientationPatient", None)
                    if iop:
                        vx = np.array(iop[0:3], dtype=np.float64)
                        vy = np.array(iop[3:6], dtype=np.float64)
                        vz = np.cross(vx, vy)
                        dose_direction = [
                            vx[0], vy[0], vz[0], vx[1], vy[1], vz[1], vx[2], vy[2], vz[2],
                        ]
                    else:
                        dose_direction = [1,0,0, 0,1,0, 0,0,1]
                    if ref_plan_uid not in plan_to_dose_info:
                        plan_to_dose_info[ref_plan_uid] = []
                    plan_to_dose_info[ref_plan_uid].append({
                        "sop": dose_sop, "spacing": dose_spacing,
                        "origin": dose_origin, "direction": dose_direction, "size": dose_size,
                    })
                except Exception:
                    pass

        # RP: one row per plan
        if not os.path.isdir(rp_dir):
            continue

        for fname in sorted(os.listdir(rp_dir)):
            if not fname.lower().endswith((".dcm", ".dicom")):
                continue
            try:
                rp_ds = pydicom.dcmread(join(rp_dir, fname), force=True)
                plan_uid = getattr(rp_ds, "SOPInstanceUID", "")
                plan_label = getattr(rp_ds, "RTPlanLabel", "") or getattr(rp_ds, "RTPlanName", "")
                patient_id = getattr(rp_ds, "PatientID", "")
                ref_struct_uid = ""
                if "ReferencedStructureSetSequence" in rp_ds:
                    ref_struct_uid = rp_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                ref_ct_series = rs_uid_to_ct_series.get(ref_struct_uid, "")
                ct_meta = ct_series_meta.get(ref_ct_series)
                if ct_meta is None and ct_series_meta:
                    ct_meta = list(ct_series_meta.values())[0]
                ct_spacing_str = json.dumps(ct_meta["spacing"]) if ct_meta else ""
                ct_origin_str = json.dumps(ct_meta["origin"]) if ct_meta else ""
                ct_direction_str = json.dumps([round(v, 6) for v in ct_meta["direction"]]) if ct_meta else ""
                ct_size_str = json.dumps(ct_meta["size"]) if ct_meta else ""
                dose_infos = plan_to_dose_info.get(plan_uid, [])
                ref_dose_uids = [d["sop"] for d in dose_infos]
                dose_meta = dose_infos[0] if dose_infos else None
                dose_spacing_str = json.dumps(dose_meta["spacing"]) if dose_meta else ""
                dose_origin_str = json.dumps(dose_meta["origin"]) if dose_meta else ""
                dose_direction_str = json.dumps([round(v, 6) for v in dose_meta["direction"]]) if dose_meta else ""
                dose_size_str = json.dumps(dose_meta["size"]) if dose_meta else ""

                # Beam parameters
                beam_names = []
                isocenters = []
                gantry_angles = []
                couch_angles = []
                collimator_angles = []
                arc_types = []
                total_cps_list = []

                if "BeamSequence" in rp_ds:
                    for b in rp_ds.BeamSequence:
                        delivery_type = getattr(b, "TreatmentDeliveryType", "TREATMENT")
                        if delivery_type != "TREATMENT":
                            continue
                        beam_names.append(getattr(b, "BeamName", ""))
                        cp0 = b.ControlPointSequence[0]
                        iso = getattr(cp0, "IsocenterPosition", None)
                        if iso:
                            isocenters.append([round(float(iso[0]), 2), round(float(iso[1]), 2), round(float(iso[2]), 2)])
                        else:
                            isocenters.append([0, 0, 0])
                        couch_angles.append(float(getattr(cp0, "PatientSupportAngle", 0) or 0))
                        collimator_angles.append(float(getattr(cp0, "BeamLimitingDeviceAngle", 0) or 0))
                        num_cps = len(b.ControlPointSequence)
                        total_cps_list.append(num_cps)
                        angle_values = []
                        current_angle = getattr(cp0, "GantryAngle", None)
                        for cp in b.ControlPointSequence:
                            ang = getattr(cp, "GantryAngle", None)
                            if ang is not None:
                                current_angle = float(ang)
                            if current_angle is not None:
                                angle_values.append(round(float(current_angle), 1))
                        if len(angle_values) >= 2:
                            gantry_angles.append([angle_values[0], angle_values[-1]])
                        elif len(angle_values) == 1:
                            gantry_angles.append([angle_values[0], angle_values[0]])
                        else:
                            gantry_angles.append([0, 0])
                        # Arc type classification
                        if len(angle_values) >= 2:
                            if abs(angle_values[0] - angle_values[-1]) < 0.5:
                                unique_angles = set(angle_values)
                                if len(unique_angles) <= 1 or (max(angle_values) - min(angle_values)) < 1.0:
                                    arc_types.append("Static")
                                else:
                                    arc_types.append("Full Arc")
                            else:
                                arc_types.append("Full Arc")
                        else:
                            arc_types.append("Static")

                row = {
                    "PatientFolder": p, "PatientID": patient_id,
                    "PlanLabel": plan_label, "PlanUID": plan_uid,
                    "RefStructUID": ref_struct_uid, "RefCTSeriesUID": ref_ct_series,
                    "CTSpacing": ct_spacing_str, "CTOrigin": ct_origin_str,
                    "CTDirection": ct_direction_str, "CTSize": ct_size_str,
                    "RefDoseUIDs": ";".join(ref_dose_uids),
                    "DoseSpacing": dose_spacing_str, "DoseOrigin": dose_origin_str,
                    "DoseDirection": dose_direction_str, "DoseSize": dose_size_str,
                    "NumTreatmentBeams": len(beam_names),
                    "BeamNames": json.dumps(beam_names), "Isocenters": json.dumps(isocenters),
                    "GantryAngles": json.dumps(gantry_angles), "CouchAngles": json.dumps(couch_angles),
                    "CollimatorAngles": json.dumps(collimator_angles), "ArcTypes": json.dumps(arc_types),
                    "TotalCPs": json.dumps(total_cps_list),
                }
                rows.append(row)
            except Exception as e:
                print(f"  Warning: Failed to read plan {fname} for {p}: {e}")

    df = pd.DataFrame(rows)
    csv_path = join(output_dir, f"{tr_ts}_dicom_index.csv")
    df.to_csv(csv_path, index=False)
    print(f"DICOM index written to {csv_path} ({len(df)} plans)")
    return df



# =============================================================================
# Stage 1-3 — Beam Export
# =============================================================================

def beam_export(
    folder,
    tr_ts="Tr",
    num_processes=1,
    use_template=False,
    use_gpu=False,
    use_field_only=False,
    structure_json=None,
):
    """
    Generate beam maps for all patients in nnDose_dicom/{name}/{tr_ts}/.

    Three modes:
      1. BEV (default): Per-control-point beam's eye view projection. Reads RP DICOM
         for each patient, generates beam map per treatment beam, sums all beams.
      2. Template (--template): Uses pre-generated templates (single_beam, half_arc, full_arc)
         shifted/rotated to isocenter from the DICOM index CSV.
      3. GPU (--gpu with BEV): Same as BEV but uses PyTorch CUDA for faster computation.

    Output: beam.nii.gz saved in each patient's Nifti/ folder.

    Args:
        folder: Base folder (e.g. nnDose_dicom/Prostate).
        tr_ts: 'Tr' or 'Ts'.
        num_processes: Number of parallel workers (for BEV mode).
        use_template: If True, use template-based beam generation.
        use_gpu: If True, use GPU (PyTorch CUDA) for BEV beam calculations.
        use_field_only: If True, use field-only mode (no divergence) for BEV.
        structure_json: Optional structure JSON used to identify target masks.
    """
    data_folder = join(folder, tr_ts)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    if use_template:
        _beam_export_template(folder, tr_ts)
    else:
        _beam_export_bev(data_folder, num_processes, use_field_only, use_gpu, structure_json)


def _beam_export_template(folder, tr_ts):
    """Template-based beam export: uses DICOM index CSV + pre-generated templates."""
    from nndosenet.formating_dataset.beam_from_template import generate_beam_for_plan_from_index

    data_folder = join(folder, tr_ts)
    index_csv = join(folder, tr_ts + "_dicom_index.csv")
    if not os.path.isfile(index_csv):
        print("DICOM index not found at %s. Run nnDoseNet_0-0_list_dicom_and_organize first." % index_csv)
        return

    df = pd.read_csv(index_csv)
    print(f"Template beam export: {len(df)} plans from {index_csv}")

    for _, plan_row in tqdm(df.iterrows(), total=len(df), desc="Beam export (template)"):
        pf = plan_row["PatientFolder"]
        nifti_dir = join(data_folder, pf, "Nifti")
        dose_path = join(nifti_dir, "dose.nii.gz")
        beam_path = join(nifti_dir, "beam.nii.gz")

        if not os.path.isfile(dose_path):
            print(f"  {pf}: No dose.nii.gz, skipping beam generation")
            continue

        try:
            dose_img = sitk.ReadImage(dose_path)
            beam_img = generate_beam_for_plan_from_index(dose_img, plan_row)
            sitk.WriteImage(beam_img, beam_path)
            print(f"  {pf}: beam.nii.gz written ({plan_row.get('NumTreatmentBeams', '?')} beams)")
        except Exception as e:
            print(f"  {pf}: WARNING beam generation failed: {e}")


def _beam_export_bev(data_folder, num_processes, use_field_only, use_gpu, structure_json=None):
    """BEV beam export: reads RP DICOM directly per patient.

    Automatically finds and combines PTV/target NIfTI files in each patient's
    Nifti/ folder to create combined_target.nii.gz for BEV beam generation.
    Target files are detected from structure_organ.json (Target category) or
    by matching PTV*.nii.gz / CTV*.nii.gz / GTV*.nii.gz patterns.
    """
    from nndosenet.formating_dataset.beam_vector import _generate_beam_for_one_plan

    patients = sorted([p for p in os.listdir(data_folder) if os.path.isdir(join(data_folder, p))])
    print(f"BEV beam export: {data_folder} ({len(patients)} patients)")
    print(f"  GPU: {use_gpu}, field_only: {use_field_only}, processes: {num_processes}")

    # Prefer an explicitly supplied structure, then try project defaults.
    base_folder = os.path.dirname(data_folder)
    target_struct_names = None
    structure_paths = []
    if structure_json is not None:
        structure_paths.append(structure_json)
    structure_paths.extend([join(base_folder, "structure_organ.json"), join(base_folder, "structure.json")])
    for json_path in structure_paths:
        if os.path.isfile(json_path):
            try:
                sj = json.load(open(json_path))
                # Look for Target category in structure_label or structure_name
                if "structure_label" in sj and "Target" in sj["structure_label"]:
                    # Get all structure names from Target category
                    target_struct_names = []
                    for label_val, names in sj["structure_label"]["Target"].items():
                        if isinstance(names, list):
                            target_struct_names.extend(names)
                        else:
                            target_struct_names.append(str(names))
                    print(f"  Target structures from {os.path.basename(json_path)}: {target_struct_names}")
                    break
            except Exception:
                pass

    for patient in tqdm(patients, desc="Beam export (BEV)"):
        patient_path = join(data_folder, patient)
        nifti_dir = join(patient_path, "Nifti")

        # Find RP file
        rp_dir = join(patient_path, "Dicom", "RP")
        if not os.path.isdir(rp_dir):
            print(f"  {patient}: No RP folder, skipping")
            continue
        rp_files = [f for f in os.listdir(rp_dir) if f.endswith('.dcm')]
        if not rp_files:
            print(f"  {patient}: No RP DICOM, skipping")
            continue

        # Build combined_target.nii.gz from available target NIfTIs
        target_path = join(nifti_dir, "combined_target.nii.gz")
        if not os.path.isfile(target_path):
            if not os.path.isdir(nifti_dir):
                print(f"  {patient}: No Nifti folder, skipping")
                continue

            # Find target files: from structure_organ.json or by PTV/CTV/GTV pattern
            nifti_files = os.listdir(nifti_dir)
            target_files = []

            if target_struct_names:
                # Match from structure JSON
                for name in target_struct_names:
                    fname = name + ".nii.gz"
                    if fname in nifti_files:
                        target_files.append(fname)

            if not target_files:
                # Fall back to pattern matching (PTV, CTV, GTV)
                for f in nifti_files:
                    f_upper = f.upper()
                    if any(pat in f_upper for pat in ["PTV", "CTV", "GTV"]):
                        target_files.append(f)

            if not target_files:
                print(f"  {patient}: No target files found (PTV/CTV/GTV), skipping BEV")
                continue

            # Combine target files into a single binary mask
            try:
                ct_path = join(nifti_dir, "ct.nii.gz")
                ref_img = sitk.ReadImage(ct_path) if os.path.isfile(ct_path) else sitk.ReadImage(join(nifti_dir, target_files[0]))
                combined = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint16)

                for tf in target_files:
                    try:
                        mask = sitk.GetArrayFromImage(sitk.ReadImage(join(nifti_dir, tf)))
                        combined = np.maximum(combined, (mask > 0).astype(np.uint16))
                    except Exception as e:
                        print(f"  {patient}: Warning, failed to read {tf}: {e}")

                combined_img = sitk.GetImageFromArray(combined)
                combined_img.CopyInformation(ref_img)
                sitk.WriteImage(combined_img, target_path)
                print(f"  {patient}: Combined {len(target_files)} target files → combined_target.nii.gz")
            except Exception as e:
                print(f"  {patient}: Failed to build combined_target: {e}")
                continue

        plan_path = join(rp_dir, rp_files[0])
        try:
            beam_img, _ = _generate_beam_for_one_plan(
                patient_path, plan_path, num_processes, use_field_only
            )
            if beam_img is not None:
                beam_path = join(nifti_dir, "beam.nii.gz")
                sitk.WriteImage(beam_img, beam_path)
                print(f"  {patient}: beam.nii.gz written")
            else:
                print(f"  {patient}: BEV beam generation returned None")
        except Exception as e:
            print(f"  {patient}: WARNING BEV beam failed: {e}")
