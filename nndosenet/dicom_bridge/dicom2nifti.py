import pydicom
import numpy as np
import SimpleITK as sitk
import os
import shutil
from rt_utils import RTStructBuilder
from nndosenet.paths import nnDose_raw,nnDose_prediction
from batchgenerators.utilities.file_and_folder_operations import *

def sitk_read_dcm(in_path,get_patient_id=True):
    #read dcm
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(in_path)
    reader.SetFileNames(dicom_names)
    dicom = reader.Execute()
    if not get_patient_id:
        return dicom, None

    # get patient id
    first_file_reader = sitk.ImageFileReader()
    first_file_reader.SetFileName(dicom_names[0])
    first_file_reader.LoadPrivateTagsOn()
    first_file_reader.ReadImageInformation()
    patient_id = first_file_reader.GetMetaData('0010|0020')

    return dicom, patient_id


class DicomConverter():#patient wise
    def __init__(self,dataset_folder,patient_folder,structure_name=None,target_prescriptions=None):
        self.dose2ct = True
        self.patient_folder = patient_folder
        self.patient_id = None
        self.target_prescriptions = target_prescriptions
        self.raw_dicom_folder = join(dataset_folder,patient_folder)

        if structure_name is None:
            self.structure_name  = load_json(join(os.path.dirname(dataset_folder),'structure.json'))["structure_name"]
        elif isinstance(structure_name,dict):
            if "structure_name" in structure_name:
                self.structure_name = structure_name["structure_name"]
            else:
                self.structure_name = structure_name
        elif isinstance(structure_name,str):
            if structure_name.endswith(".json") and isfile(structure_name):
                self.structure_name = load_json(structure_name)["structure_name"]
            else:
                raise ValueError('the structure.json does not exist or it is not end wih json, please check the input again! ')

        self.ct_folder = join(self.raw_dicom_folder,"Dicom","CT")
        self.dose_folder = join(self.raw_dicom_folder,"Dicom","RD")
        self.struct_folder = join(self.raw_dicom_folder,"Dicom","RS")
        self.plan_folder = join(self.raw_dicom_folder,"Dicom","RP")
        self.nii_folder = join(self.raw_dicom_folder,"Nifti")


    def run(self):
        _, self.patient_id = self.get_ct(save=True, patient_id_dicom=True,overwrite=False)

        self.get_dose()
        structure_found = self.confirm_structure_and_2nii()
        if self.target_prescriptions is not None:
            self.create_combined_target(self.target_prescriptions)
        return structure_found

#TODO: change this
    def get_ct(self,save=False,patient_id_dicom=False,overwrite=False):

        ct_path = join(self.nii_folder,"ct.nii.gz")
        if os.path.isfile(ct_path) and not overwrite:
            patient_id = None
            if patient_id_dicom:
                ct_files = sorted(os.listdir(self.ct_folder))
                if ct_files:
                    patient_id = str(getattr(
                        pydicom.dcmread(join(self.ct_folder, ct_files[0]), stop_before_pixels=True),
                        "PatientID", "",
                    )) or None
            return sitk.ReadImage(ct_path), patient_id

        # from dicom
        if len (os.listdir(self.ct_folder))>0:
            ct, patient_id = sitk_read_dcm(self.ct_folder,get_patient_id=patient_id_dicom)
            if save:
                # print("write",ct_path)
                sitk.WriteImage(ct,ct_path)
                # sitk.WriteImage(ct,join(self.nii_folder,"img.nii.gz"))
                # print("done")
            return ct, patient_id

        # #form nifti img.nii.gz
        # if os.path.isfile(join(self.nii_folder,"img.nii.gz")):
        #     print('"ct.nii.gz" is not in Nifti folder but found "img.nii.gz".\n'\
        #             'I am assuming "img.nii.gz" is the ct or at least the img you want to use as input\n'\
        #             'so I am Renaming img.nii.gz to ct.nii.gz....\n'\
        #             'if this is not what you want.... please start over and see the instruction carefully. Thank you!')
        #     os.rename(join(self.nii_folder,"img.nii.gz"),join(self.nii_folder,"ct.nii.gz"))
        #     return sitk.ReadImage(ct_path)

        raise RuntimeError("If you are using Nifti, please name ct in ct.nii.gz in Nifti folder")

    def get_dose(self):
        dose_path = join(self.nii_folder,"dose.nii.gz")
        # if os.path.isfile(dose_path):
        #     return sitk.ReadImage(dose_path)

        if len(os.listdir(self.dose_folder)) == 1:
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose = pydicom.dcmread(join(self.dose_folder,dose_path))
                return self.convert_dose2nii(dose,out_path=join(self.nii_folder,"dose.nii.gz"))

        if(len(os.listdir(self.dose_folder)) > 1):
            for i,dose_path in enumerate(os.listdir(self.dose_folder)):
                dose = pydicom.dcmread(join(self.dose_folder,dose_path))
                print("WARNING! Multi Dose Plan Detected: only converting dose: ",dose_path," in",self.raw_dicom_folder, )
                return self.convert_dose2nii(dose,out_path=join(self.nii_folder,"dose.nii.gz"))

        raise RuntimeError("can't even find the dicom for dose, please organize everything as instruction")

    def convert_dose2nii(self,dose_data,out_path=None):
        if out_path == None:
            out_path = join(self.nii_folder,"dose.nii.gz")

        ct,_ = self.get_ct(save=False,patient_id_dicom=False,overwrite=False)

        dose = dose_data.pixel_array.astype(float)
        dose = dose*float(dose_data.DoseGridScaling)

        dose = sitk.GetImageFromArray(dose)
        dose.SetOrigin(dose_data.ImagePositionPatient)
        grid_frame_offset_vector = dose_data.GridFrameOffsetVector  # List of Z positions
        z_spacing = float(grid_frame_offset_vector[1] - grid_frame_offset_vector[0])

        dose.SetSpacing([dose_data.PixelSpacing[0],dose_data.PixelSpacing[1],z_spacing])
        dose.SetDirection(ct.GetDirection())
        # if float(ct.GetSpacing()[2]) != z_spacing:
        #     print("oh no, I am really screw up  ",self.nii_folder," ct, z",float(ct.GetSpacing()[2]), z_spacing)

        # nnDoseNet requires every input channel and the dose label to share the
        # CT grid. RTDOSE grids commonly differ in in-plane resolution, extent,
        # and origin, so preserve dose values with linear interpolation while
        # resampling onto CT (outside-dose-grid voxels are zero dose).
        if self.dose2ct:
            dose = sitk.Resample(
                dose,
                ct,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32,
            )

        sitk.WriteImage(dose,out_path)

    def resample_to_ct(self,image: sitk.Image,transform=sitk.Euler3DTransform(),
                 interpolator = sitk.sitkNearestNeighbor,default_value=0):
        reference_image,_ = self.get_ct(save=False,patient_id_dicom=False,overwrite=False)
        return sitk.Resample(image, reference_image, transform,
                                interpolator, default_value)

    def confirm_structure_and_2nii(self):
        ''' search and "only" extracting "first" found candidate_roi_name in each defined OAR by order
            e.g. structure_name:{"Femur_Head": ["Femur_Head_L","Femur_Head_R"]}.
            We are "only" going to use "Femur_Head_L" if it exist, if not we would search for "Femur_Head_L .
            If we can't find both, the mask would be left empty in this stage.

            If you want both "Femur_Head_L" and "Femur_Head_R", please formate like this
            structure_name:={"Femur_Head_L": ["Femur_Head_L"],
                                "Femur_Head_R": ["Femur_Head_R"]}

            And if want them to combine as 1 structure, you can label them as same value, like this
            structure_label:{"OAR": {"50":["Femur_Head_L","Femur_Head_R"]}
        '''
        ct,_ = self.get_ct(save=False,patient_id_dicom=False,overwrite=False)
        # Target prescriptions belong to the patient-specific target summary,
        # rather than the shared organ mapping. Add their exact ROI names here
        # so those masks are always available for combined_target generation.
        structure_map = dict(self.structure_name)
        if self.target_prescriptions is not None:
            if not isinstance(self.target_prescriptions, dict) or not self.target_prescriptions:
                raise ValueError(f"Target summary for {self.patient_folder} must be a non-empty object")
            for target_name in self.target_prescriptions:
                if not isinstance(target_name, str) or not target_name or os.path.basename(target_name) != target_name:
                    raise ValueError(f"Invalid target ROI name {target_name!r} for {self.patient_folder}")
                # The R1 spelling occurs in some RTSTRUCTs. Write either match
                # using the summary's primary target name as the NIfTI filename.
                structure_map.setdefault(target_name, [target_name, target_name + "^R1"])

        structure_list = list(structure_map.keys())
        structure_found_dict = {}
        structure_found_dict.update({'patient_id':self.patient_id})
        for structure in structure_list:
            structure_found_dict.update({structure:""})
        structure_found_dict.update({'not_matched':[]})

        for struct_file in os.listdir(self.struct_folder):
            # Load existing RT Struct. Requires the series path and existing RT Struct path
            rtstruct = RTStructBuilder.create_from(
                                        dicom_series_path=self.ct_folder,
                                        rt_struct_path=join(self.struct_folder,struct_file))
            roi_list = rtstruct.get_roi_names()
            structure_found_dict['not_matched']=structure_found_dict['not_matched']+list(roi_list)

            for structure in structure_list:
                for name in structure_map[structure]:
                    if name in roi_list:
                        try:
                            # Get the mask
                            mask_array = np.transpose(rtstruct.get_roi_mask_by_name(name)*int(1), axes=[2,0,1]).astype(np.int32)
                            mask = sitk.GetImageFromArray(mask_array)

                            # # Check if sizes match
                            # if mask.GetSize() != ct.GetSize():
                            #     print(f"Resampling mask of  {self.patient_folder}{self.patient_id}  for {name} from {mask.GetSize()} to {ct.GetSize()}")
                            #     # Create resampler
                            #     mask.SetOrigin(ct.GetOrigin())
                            #     mask.SetSpacing(ct.GetSpacing())
                            #     mask.SetDirection(ct.GetDirection())
                            #     resampler = sitk.ResampleImageFilter()
                            #     resampler.SetReferenceImage(ct)
                            #     resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for binary masks
                            #     resampler.SetDefaultPixelValue(0)
                            #     # Resample the mask
                            #     mask = resampler.Execute(mask)

                            mask.CopyInformation(ct)
                            sitk.WriteImage(mask,join(self.nii_folder,structure+".nii.gz"))
                            structure_found_dict[structure]=name
                            # The same ROI can legitimately be requested by a
                            # shared structure map and a target summary. Remove
                            # it only once from the diagnostic list.
                            if name in structure_found_dict['not_matched']:
                                structure_found_dict['not_matched'].remove(name)
                            break
                        except Exception as e:
                            print(f"Error processing {self.patient_folder}{self.patient_id} {name}: {str(e)}")
                            continue

        return structure_found_dict

    def create_combined_target(self, target_prescriptions):
        """Write ``combined_target.nii.gz`` from the patient target summary.

        Values are prescription Gy multiplied by 100, matching the established
        process-code convention. Target masks are applied from lower to higher
        prescription so that higher-dose targets own overlapping voxels.
        """
        ct, _ = self.get_ct(save=False, patient_id_dicom=False, overwrite=False)
        shape = sitk.GetArrayFromImage(ct).shape
        combined = np.zeros(shape, dtype=np.float32)
        used_targets = []

        try:
            sorted_targets = sorted(target_prescriptions.items(), key=lambda item: (float(item[1]), item[0]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Target-summary prescriptions for {self.patient_folder} must be numeric Gy values"
            ) from e

        for target_name, prescription_gy in sorted_targets:
            prescription_gy = float(prescription_gy)
            if not np.isfinite(prescription_gy) or prescription_gy <= 0:
                raise ValueError(
                    f"Target-summary prescription for {self.patient_folder}/{target_name} must be positive and finite"
                )

            target_path = join(self.nii_folder, target_name + ".nii.gz")
            if not os.path.isfile(target_path):
                raise FileNotFoundError(
                    f"Target ROI {target_name!r} from the target summary was not found in "
                    f"the RTSTRUCT for {self.patient_folder}"
                )

            target_mask = sitk.GetArrayFromImage(sitk.ReadImage(target_path)) > 0
            if target_mask.shape != shape:
                raise RuntimeError(
                    f"Target-mask shape mismatch for {target_path}: {target_mask.shape} != CT shape {shape}"
                )
            if np.any(target_mask):
                combined[target_mask] = prescription_gy * 100.0
                used_targets.append(target_name)

        if not used_targets:
            raise RuntimeError(f"No non-empty target masks were found for {self.patient_folder}")

        combined_image = sitk.GetImageFromArray(combined)
        combined_image.CopyInformation(ct)
        sitk.WriteImage(combined_image, join(self.nii_folder, "combined_target.nii.gz"))

class NiftiLabeler(DicomConverter):

    def __init__(self, folder, patient_folder, patient_id, dataset_name, dataset_id, tr_ts = "Tr", structure_label=None,
                 target_summary_path=None):

        self.dataset_name = dataset_name
        self.patient_folder = patient_folder
        self.patient_id = str(patient_id).zfill(4) #FIXME: this might caus the space in the patient_id
        #print(self.patient_id)
        self.tr_ts=tr_ts
        self.use_combined_target = target_summary_path is not None


        self.raw_dicom_folder = join(folder,patient_folder)

        if tr_ts=="Tr":
            self.raw_dataset_folder = join(nnDose_raw,self.dataset_name)
        else:
            dataset_name_only=""
            for str_split in self.dataset_name.split("_"):
                if "Dataset" in str_split:
                    continue
                dataset_name_only=dataset_name_only+"_"+str_split
            dataset_name_only=dataset_name_only[1:]
            self.raw_dataset_folder = join(nnDose_prediction,dataset_name_only,self.dataset_name,"GroundTruth")

        if not os.path.isdir(self.raw_dataset_folder):
            os.makedirs(self.raw_dataset_folder)
            print("Create folder in : ",self.raw_dataset_folder)

        if structure_label is None:
            self.structure_label  = load_json(join(os.path.dirname(folder),'structure.json'))["structure_label"]
        elif isinstance(structure_label,dict):
            if "structure_label" in structure_label:
                self.structure_label = structure_label["structure_label"]
            else:
                self.structure_label = structure_label
        elif isinstance(structure_label,str):
            if structure_label.endswith(".json") and isfile(structure_label):
                self.structure_label = load_json(structure_label)["structure_label"]
            else:
                raise ValueError('the structure.json does not exist or it is not end wih json, please check the input again! ')


        #print(structure_label)
        self.nii_folder=join(self.raw_dicom_folder,"Nifti")

    def run(self):
        self.confirm_and_label()

    def confirm_and_label(self):
        ct,_=self.get_ct(save=False,patient_id_dicom=False,overwrite=False)
        #Copy dose and ct
        shutil.copy2(join(self.nii_folder,"ct.nii.gz"),join(self.raw_dataset_folder,"images"+self.tr_ts,self.patient_id+"_0000.nii.gz"))
        shutil.copy2(join(self.nii_folder,"dose.nii.gz"),join(self.raw_dataset_folder,"labels"+self.tr_ts,self.patient_id+".nii.gz"))
        shape=sitk.GetArrayFromImage(ct).shape
        def make_labeled_mask(category, default_to_full_image=False):
            """Combine the configured ROI masks for one nnDoseNet input channel.

            Values in ``structure_label`` are dose/prescription values for targets
            and clinical-goal values for OARs, so they may be non-integers. Later
            values deliberately take precedence in overlaps, matching the former
            labeling behaviour.
            """
            image = np.zeros(shape, dtype=np.float32)
            category_labels = self.structure_label.get(category, {})
            found_mask = False

            def sort_key(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float("inf")

            for value in sorted(category_labels, key=sort_key):
                try:
                    label_value = float(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"{category} label value {value!r} for {self.patient_folder} must be numeric"
                    ) from e

                structures = category_labels[value]
                if not isinstance(structures, list):
                    structures = [structures]
                mask = np.zeros(shape, dtype=bool)
                for structure in structures:
                    path = join(self.nii_folder, str(structure) + ".nii.gz")
                    if os.path.isfile(path):
                        roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0
                        if roi_mask.shape != shape:
                            raise RuntimeError(
                                f"Mask shape mismatch for {path}: {roi_mask.shape} != CT shape {shape}"
                            )
                        mask |= roi_mask
                if np.any(mask):
                    image[mask] = label_value
                    found_mask = True

            if default_to_full_image and not found_mask:
                image.fill(1.0)
            return image, found_mask

        output_channels = (("Target", "_0001", False), ("OAR", "_0002", False), ("Body", "_0003", True))
        for category, suffix, default_to_full_image in output_channels:
            if category == "Target" and self.use_combined_target:
                combined_target_path = join(self.nii_folder, "combined_target.nii.gz")
                if not os.path.isfile(combined_target_path):
                    raise FileNotFoundError(
                        f"Target summary was requested but {combined_target_path} does not exist. "
                        "Run DICOM conversion with the same -target_summary first."
                    )
                image_sitk = sitk.ReadImage(combined_target_path)
                image = sitk.GetArrayFromImage(image_sitk)
                if image.shape != shape:
                    raise RuntimeError(
                        f"Combined-target shape mismatch for {combined_target_path}: {image.shape} != CT shape {shape}"
                    )
                if not np.any(image > 0):
                    raise RuntimeError(f"Combined target is empty for {self.patient_folder}")
                found_mask = True
            else:
                image, found_mask = make_labeled_mask(category, default_to_full_image)
                image_sitk = sitk.GetImageFromArray(image)
            if category == "Target" and not found_mask:
                print(f"WARNING: No configured target mask found for {self.patient_folder}; writing an empty Target channel")
            if category == "Body" and not found_mask:
                print(f"WARNING: No configured Body mask found for {self.patient_folder}; using a full-image Body mask")
            image_sitk.CopyInformation(ct)
            sitk.WriteImage(image_sitk, join(self.raw_dataset_folder, "images" + self.tr_ts, self.patient_id + suffix + ".nii.gz"))

#
