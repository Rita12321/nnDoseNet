import numpy as np 

class DoseEvaluator:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics"""

    def __init__(self, dose_ref, dose_pred, seg, label_df,metrics_list=None,voxel_size=1,seperate_label=False):
        self.dose_ref = dose_ref
        self.dose_pred = dose_pred
        self.seg = seg
        self.label_df = label_df
        self.voxel_size = voxel_size
        self.roi_list = self.get_roi_dict(seperate_label)
        if metrics_list is None:
            self.metrics = ["mean","min","max","D_0.1_cc","D_99","D_95","D_1","homogeneity","conformity"]
        else:
            self.metrics=metrics_list
        self.calculate_dvh_metrics(seperate_label)
        

    def get_roi_dict(self,seperate_label):
        roi={}
        for label in self.label_df:
            temp_str=""
            for structure in self.label_df[label]:
                if seperate_label:
                    roi.update({structure:{"input_label":label}})
                else:
                    temp_str=temp_str+"+"+structure
            if not seperate_label:
                roi.update({temp_str[1:]:{"input_label":label}})
        return roi   

    def calculate_dvh_metrics(self,seperate_label=False):
        """
        Calculate the DVH values that were used to evaluate submissions in the competition.
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        voxels_within_tenths_cc = np.maximum(1, np.round(100 / self.voxel_size))
        for i,structure in enumerate(self.roi_list):
            temp_dict=self.roi_list[structure]
            label=self.roi_list[structure]["input_label"]
            if seperate_label or (self.seg is list):
                
                masked_ref=np.where(self.seg[i]>0,self.dose_ref,np.nan)
                masked_ref = masked_ref[np.logical_not(np.isnan(masked_ref))]
                if len(masked_ref)<=0: 
                    continue
                masked_pred=np.where(self.seg[i]>0,self.dose_pred,np.nan) 
                masked_pred = masked_pred[np.logical_not(np.isnan(masked_pred))]
            else:
                masked_ref=np.where(self.seg==int(label),self.dose_ref,np.nan)
                masked_ref = masked_ref[np.logical_not(np.isnan(masked_ref))]
                if len(masked_ref)<=0: 
                    continue
                masked_pred=np.where(self.seg==int(label),self.dose_pred,np.nan) 
                masked_pred = masked_pred[np.logical_not(np.isnan(masked_pred))]
                
            
            for metric in self.metrics:
               # print(metric)
                for pred_ref in ['Pred','Ref']:
                    if pred_ref =='Pred':
                        roi_dose=masked_pred
                    else:
                        roi_dose=masked_ref
                    roi_size = len(roi_dose)
                    if roi_size<=0:
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
                        metric_value == (np.percentile(roi_dose, 98)-np.percentile(roi_dose, 2))/np.percentile(roi_dose, 50)
                    elif metric == "homogeneity":
                        metric_value == 0#(np.percentile(roi_dose, 98)-np.percentile(roi_dose, 2))/np.percentile(roi_dose, 50)      
                    else:
                        raise ValueError(f"Metrics {metric} is not supported.")
                    temp_dict.update({metric+"_"+pred_ref:metric_value})
             
                   
            self.roi_list.update({structure:temp_dict})
          

        return self.roi_list





