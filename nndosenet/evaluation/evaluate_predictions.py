import multiprocessing
import os
import pandas as pd 
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional
from sklearn.metrics import mean_squared_error as MSE
# from pymedphys import gamma as gamma_analysis
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nndosenet.configuration import default_num_processes
from nndosenet.imageio.base_reader_writer import BaseReaderWriter
from nndosenet.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nndosenet.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnDoseNet was great and all but man was it overengineered. Keep it simple
from nndosenet.utilities.json_export import recursive_fix_for_json_export
# from nndosenet.utilities.plans_handling.plans_handler import PlansManager
from nndosenet.evaluation.basic_DVH_metrics import DoseEvaluator
from nndosenet.paths import nnDose_dicom



def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return key
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        splitted = key.split(',')
        return tuple([i for i in splitted])


def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    #print(results_converted) 
    # convert keys in mean metrics
    try:
        results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    except:
        print("no means")
    # convert metric_per_case 
    try:
        for i in range(len(results_converted["metric_per_case"])):
        
            results_converted["metric_per_case"][i]['metrics'] = \
                {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
                for k in results["metric_per_case"][i]['metrics'].keys()}
    except:
        print("no metric_per_case")
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=False)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    try:
        results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    except:
        print("no mean")
    # convert metric_per_case
    try:
        for i in range(len(results["metric_per_case"])):
            results["metric_per_case"][i]['metrics'] = \
                {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
                for k in results["metric_per_case"][i]['metrics'].keys()}
    except:
        print("no metric_per_case")
    return results
def compute_dose_metrics(reference_file: str, prediction_file: str, ptv_file: str, oar_file: str,
                         image_reader_writer: BaseReaderWriter,
                        structure_label: Union[List[int], List[Union[int, Tuple[int, ...]]]]) -> dict:
    
    PTV_METRICS=["mean","min","max","D_0.1_cc","D_99","D_95","D_1","homogeneity","conformity"]#
    OAR_METRICS=["mean","min","max","D_0.1_cc"]
    # metrics: full image evaluation, only need ref and pred
    dose_ref, dose_dict = image_reader_writer.read_gt(reference_file)
    dose_pred, _ = image_reader_writer.read_gt(prediction_file)
    dose_ref = dose_ref.squeeze().flatten()
    dose_pred = dose_pred.squeeze().flatten()    
    
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['target_file/folder'] = ptv_file
    results['OAR_file/folder'] = oar_file
    results['dose'] = 'True'
    results['prescription']= -1
    results['metrics']={}
    results['metrics']['MSE'] = MSE(dose_ref,dose_pred)
    results['metrics']['max_Ref'] = np.max(dose_ref)
    results['metrics']['max_Pred'] = np.max(dose_pred)
    # DVH: ROI/contour/seg evaluation, need ref, pred and cotour  
    shape=dose_ref.shape
    voxel_size = np.prod(dose_dict['sitk_stuff']['spacing']) 
    if os.path.isdir(ptv_file) :
        seg_ptv=[]

        for i in structure_label['Target'].values():
            for j in i:
                path=join(ptv_file,j+".nii.gz")
                if os.path.isfile(path):
                    temp, _ = image_reader_writer.read_gt(path)
                    seg_ptv.append(temp.squeeze().flatten())
                    
                else:
                    seg_ptv.append(np.zeros(shape)) 
        results['Target'] = DoseEvaluator(dose_ref, dose_pred, seg_ptv, 
                                structure_label['Target'],voxel_size,metrics_list=PTV_METRICS,separate_label=True).calculate_dvh_metrics()
    else:
        seg_ptv, _ = image_reader_writer.read_gt(ptv_file)
        seg_ptv = seg_ptv.squeeze().flatten()
        results['Target'] = DoseEvaluator(dose_ref, dose_pred, seg_ptv, 
                                structure_label['Target'],voxel_size,metrics_list=PTV_METRICS,separate_label=False).calculate_dvh_metrics() 
    if os.path.isdir(oar_file) :
        seg_oar=[]
        for i in structure_label['OAR'].values():
            for j in i:
                path=join(oar_file,j+".nii.gz")
                if os.path.isfile(path):
                    temp, _ = image_reader_writer.read_gt(path)
                    seg_oar.append(temp.squeeze().flatten())
                else:
                    seg_oar.append(np.zeros(shape)) 
        results['OAR'] = DoseEvaluator(dose_ref, dose_pred, seg_oar, 
                                structure_label['OAR'],voxel_size,metrics_list=OAR_METRICS,separate_label=True).calculate_dvh_metrics()
    else:
        seg_oar, _ = image_reader_writer.read_gt(oar_file)
        seg_oar = seg_oar.squeeze().flatten()
        results['OAR'] = DoseEvaluator(dose_ref, dose_pred, seg_oar, 
                                structure_label['OAR'],voxel_size,metrics_list=OAR_METRICS,separate_label=False).calculate_dvh_metrics()


    return results



def compute_dose_metrics_on_folder(folder_ref: str, folder_pred: str, 
                                   folder_input:str, output_file: str,
                                   image_reader_writer: BaseReaderWriter,
                                   file_ending: str,
                                   structure_df: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                   ignore_label: int = None,
                                   dose_label:bool=True,
                                   num_processes: int = 8,
                                   chill: bool = True,ptv_oar=[1,2],raw_folder=False,patient_csv=None) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
        

    if raw_folder:
        files_ptv = [join(folder_input, i,"Nifti") for i in files_pred]
        files_oar = files_ptv.copy()
        if not os.path.isfile(files_ptv[0]):
            if not os.path.isfile(patient_csv):
                raise(RuntimeError("cannot find", files_ptv[0]," and ", patient_csv,"need patient and id conversion csv to read the file from nndose_dicom"))
            csv=pd.read_csv(patient_csv)
            p_id=dict(zip (csv['id'],csv['patient']))

            files_ptv = [join(folder_input,p_id[ int(id[0:-len(file_ending)])],"Nifti") for i, id in enumerate(files_pred)]
            files_oar =files_ptv.copy()
    else:
        files_ptv = [join(folder_input, i[0:-len(file_ending)]+"_000"+str(ptv_oar[0])+file_ending) for i in files_pred]
        files_oar = [join(folder_input, i[0:-len(file_ending)]+"_000"+str(ptv_oar[1])+file_ending) for i in files_pred]
    
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:


        
        print("computing dose metrics")
        results = pool.starmap(
            compute_dose_metrics,
            list(zip(files_ref, files_pred, files_ptv, files_oar, [image_reader_writer] * len(files_pred),
                        [structure_df] * len(files_pred),
                    ))
        )


    image_prescription_compare_list=['max_Ref','max_Pred']
    target_prescription_compare_list=['D_99_Pred','D_99_Ref','D_95_Pred','D_95_Ref']
    means={}
    # find max prescription per case 
    for i in results:
        for c in i['Target'].keys():
            if i['prescription'] < float(i['Target'][c]["input_label"]):
                i['prescription'] = float(i['Target'][c]["input_label"])
    
    # mean image metrices 
    metric_list = results[0]['metrics'].keys()
    for m in metric_list:
        count_107 = 0
        count_105 = 0
        for i in results:
            means[m] = np.nanmean(i['metrics'][m] )
            if m in image_prescription_compare_list:
                means[m]
                if i['metrics'][m] < i['prescription']*1.07:
                    count_107 += 1
                    count_105 += 1
        if m in image_prescription_compare_list:
            means["# < 105% prescription of "+m] = count_105
            means["# < 107% prescription of "+m] = count_107
            
    # mean contour metrics 
    for typ in ["Target","OAR"]:
        means[typ]={} 
        contour_list = list(results[0][typ].keys())      
        metric_list = results[0][typ][contour_list[0]].keys()#
        for c in contour_list:
            means[typ][c]={}
            means[typ][c]["input_label"]=results[0][typ][c]["input_label"]

            for m in metric_list:
                count=0
                if m =="input_label":
                    continue
                temp_list=[]
                
                for r in results:
                    try:
                        if m in target_prescription_compare_list:
                            if r[typ][c][m] > float(means[typ][c]["input_label"]):
                                count += 1
                
                        temp_list.append(r[typ][c][m])
                    except:
                        pass
        
                means[typ][c][m]=np.mean(temp_list)
                
                if m in target_prescription_compare_list:
                    means[typ][c]["# > target prescription of "+m] = count
                    
                
    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    
    
    result = {'mean': means,'metric_per_case': results}


    if output_file is not None:
        save_summary_json(result, output_file)
    return result


    

def compute_dose_metrics_on_folder2(folder_ref: str, folder_pred: str,folder_input:str,
                                dataset_json_file: str,# plans_file: str, 
                                structure_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,raw_folder=False,patient_csv=None):
    dataset_json = load_json(dataset_json_file)
    try:
        structure_json= load_json(structure_file)['structure_label']
    except:
        structure_json=None
        print("Warning: Error while reading structure label !")
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')


    compute_dose_metrics_on_folder(folder_ref, folder_pred, folder_input, output_file, 
                                   rw, file_ending,structure_json,
                                   num_processes=num_processes,ignore_label=None,dose_label=True, chill=chill,
                              raw_folder=raw_folder,patient_csv=patient_csv)

    
def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with ground truth label')
    parser.add_argument('pred_folder', type=str, help='folder with predicted label')
    parser.add_argument('image_folder', type=str, help='folder with input data, that contain nifti contours')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-id_patient_csv', type=str, default=None, help='csv file of patient and id list ')
    parser.add_argument('-sjfile', type=str, required=False,
                        help='structure.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred doesnt have all files that are present in folder_gt')
    parser.add_argument('--by_roi', action='store_true', help='enable this for image_folder is nnDose_dicom format and evaluate by contours not by label')
    args = parser.parse_args()
    if args.by_roi and not(nnDose_dicom in args.image_folder):
        print("detected that image_folder is not under nndosenet dicom")
        print("with --by_roi flag, the image folder should struct as folder in nndose_dicom, that \
            that should be [image_folder]/[Nifti]/oar1.nii.gz, oar2.nii.gz, ptv1.nii.gz....continue ")
    compute_dose_metrics_on_folder2(args.gt_folder, args.pred_folder, args.image_folder, args.djfile, args.sjfile, args.o, 
                                    args.np, chill=args.chill,raw_folder=args.by_roi,patient_csv=args.id_patient_csv)
    

if __name__ == '__main__':
    #I know I spelled prediction wrong when i am naming my project folder....
    #but I have too many code using abs path that i can't changed it.Please don't laugh at me ><
    folder_ref = '/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/openkbp/Dataset024/GroundTruth/labelsTs'
    folder_pred = '/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/openkbp/Dataset024/Prediction'
    folder_input = '/home/changh/Documents/2023_DosePredcition/nnDose_1_dicom/openkbp/Ts'
    output_file = '/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/openkbp/Dataset024/summary.json'
    structure_json = load_json('/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/openkbp/Dataset024/Prediction/structure.json')['structure_label']
    patient_csv = "/home/changh/Documents/2023_DosePredcition/nnDose_5_prediction/openkbp/Dataset024/GroundTruth/patientTs.csv"
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'

    ignore_label = None
    dose_label= True
    num_processes = 12
    compute_dose_metrics_on_folder(folder_ref, folder_pred, folder_input, output_file,
                              image_reader_writer,
                              file_ending,
                              structure_json,
                              ignore_label,
                              dose_label,
                              num_processes,raw_folder=True,patient_csv=patient_csv)
    
 