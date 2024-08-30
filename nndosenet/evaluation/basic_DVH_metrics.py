import numpy as np 

class DoseEvaluator:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics and some extension metrices we add up """

    def __init__(self, dose_ref, dose_pred, seg, label_df,voxel_size = 1,metrics_list = None,separate_label = False):
        self.dose_ref = dose_ref
        self.dose_pred = dose_pred
        self.seg = seg
        self.label_df = label_df
        self.voxel_size = voxel_size
        self.separate_label = separate_label
        self.roi_list = self.get_roi_dict(separate_label)
        if metrics_list is None:
            self.metrics = ["mean","min","max","D_0.1_cc","D_99","D_95","D_1","homogeneity","conformity"]
        else:
            self.metrics = metrics_list
        

    def get_roi_dict(self,separate_label):
        roi={}
        for label in self.label_df:
            temp_str=""
            for structure in self.label_df[label]:
                if separate_label:
                    roi.update({structure:{"input_label":label}})
                else:
                    temp_str = temp_str+"+"+structure
            if not separate_label:
                roi.update({temp_str[1:]:{"input_label":label}})
        return roi   

    def calculate_dvh_metrics(self):
        """
        Calculate some of common DVH values.
        """
        voxels_within_tenths_cc = np.maximum(1, np.round(100 / self.voxel_size))
        for i,structure in enumerate(self.roi_list):
            temp_dict = self.roi_list[structure]
            label = self.roi_list[structure]["input_label"]
            if self.seperate_label or (self.seg is list):
                
                masked_ref = np.take(self.dose_ref,np.where(self.seg[i]>0)).flatten()
                if len(masked_ref) <= 0: 
                    continue
                masked_pred = np.take(self.dose_pred,np.where(self.seg[i]>0)).flatten() 
            else:
                
                masked_ref = np.take(self.dose_ref,np.where(self.seg == int(label))).flatten()
                if len(masked_ref) <= 0: 
                    continue
                masked_pred = np.take(self.dose_pred,np.where(self.seg == int(label))).flatten()
            

                
            
            for metric in self.metrics:
               # print(metric)
                for pred_ref in ['Pred','Ref']:
                    if pred_ref =='Pred':
                        roi_dose = masked_pred
                        dose = self.dose_pred
                    else:
                        roi_dose=masked_ref
                        dose=self.dose_ref
                    roi_size = len(roi_dose)
                    if roi_size <= 0:
                        continue
                    if metric == "D_0.1_cc": 
                        fractional_volume_to_evaluate = 100 - voxels_within_tenths_cc / roi_size * 100
                        metric_value = np.percentile(roi_dose, fractional_volume_to_evaluate)
                    elif metric == "mean":
                        metric_value = roi_dose.mean()
                    elif metric == "D_99":
                        metric_value = np.percentile(roi_dose, 1)
                    elif metric == "D_95":
                        metric_value = np.percentile(roi_dose, 5)
                    elif metric == "D_1":
                        metric_value = np.percentile(roi_dose, 99)
                    elif metric == "min":
                        metric_value = np.min(roi_dose)
                    elif metric == "max":
                        metric_value = np.max(roi_dose)
                    elif metric == "homogeneity":
                        metric_value = (np.percentile(roi_dose, 98)-np.percentile(roi_dose, 2))*100/float(label)
                    elif metric == "conformity":
                        V_iso = np.count_nonzero(dose >= int(label))
                        V_ptv_and = np.count_nonzero(roi_dose >= int(label))
                        if V_iso <= 0:
                            continue
                        metric_value = (V_ptv_and**2)*100/(V_iso*roi_size)
                    else:
                        raise ValueError(f"Metrics {metric} is not supported.")
                    temp_dict.update({metric+"_"+pred_ref:metric_value})
             
                   
            self.roi_list.update({structure:temp_dict})
          

        return self.roi_list





