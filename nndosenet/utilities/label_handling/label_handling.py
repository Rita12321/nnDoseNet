from __future__ import annotations
from time import time
from typing import Union, List, Tuple, Type

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import join

import nndosenet
from nndosenet.utilities.find_class_by_name import recursive_find_python_class

from typing import TYPE_CHECKING

# see https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from nndosenet.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class LabelManager(object):
    def __init__(self, label_dict: dict, regions_class_order: Union[List[int], None], force_use_labels: bool = False,
                 dose_label:bool = False, inference_nonlin=None):
        self._sanity_check(label_dict)
        self.label_dict = label_dict
        self.dose_label = dose_label
             
        self.regions_class_order = regions_class_order
        self._force_use_labels = force_use_labels

        self._all_labels: List[int] = self._get_all_labels()


    ## it only check if there is label back ground.
    def _sanity_check(self, label_dict: dict):
        if not 'background' in label_dict.keys():
            raise RuntimeError('Background label not declared (remeber that this should be label 0!)')
        bg_label = label_dict['background']
        if isinstance(bg_label, (tuple, list)):
            raise RuntimeError(f"Background label must be 0. Not a list. Not a tuple. Your background label: {bg_label}")
        assert int(bg_label) == 0, f"Background label must be 0. Your background label: {bg_label}"
        # not sure if we want to allow regions that contain background. I don't immediately see how this could cause
        # problems so we allow it for now. That doesn't mean that this is explicitly supported. It could be that this
        # just crashes.
        
    ## NOTE: changed: add 'dose' condition
    def _get_all_labels(self) -> List:#List[int]
        all_labels = []
        for k, r in self.label_dict.items():
            # ignore label is not going to be used, hence the name. Duh.
            if k == 'ignore':
                continue
            if k == 'dose':
                self.dose_label=True
                continue
            
            if isinstance(r, (tuple, list)):
                for ri in r:
                    all_labels.append(int(ri))
            else:
                all_labels.append(int(r))
                
        if self.dose_label!=True:
            self.dose_label=False
            
        all_labels = list(np.unique(all_labels))
        all_labels.sort()
        
        return all_labels

 
    @property
    def all_labels(self) -> List[int]:
        return self._all_labels


    def probability_squeeze_channel(self, predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:

        
        if isinstance(predicted_probabilities,  torch.Tensor):
            predicted_probabilities.cpu().numpy()
        if predicted_probabilities.dtype != np.float32:
            predicted_probabilities = predicted_probabilities.astype(np.float32)
            
        return predicted_probabilities[0].squeeze()

    def convert_logits_to_image_array(self, predicted_logits: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]: 
                
        return self.probability_squeeze_channel(predicted_logits)

    def revert_cropping_on_probabilities(self, predicted_probabilities: Union[torch.Tensor, np.ndarray],
                                         bbox: List[List[int]],
                                         original_shape: Union[List[int], Tuple[int, ...]]):
        """
        ONLY USE THIS WITH PROBABILITIES, DO NOT USE LOGITS AND DO NOT USE FOR SEGMENTATION MAPS!!!

        predicted_probabilities must be (c, x, y(, z))

        Why do we do this here? Well if we pad probabilities we need to make sure that convert_logits_to_image_array
        correctly returns background in the padded areas. Also we want to ba able to look at the padded probabilities
        and not have strange artifacts.
        Only LabelManager knows how this needs to be done. So let's let him/her do it, ok?
        """
        # revert cropping
        probs_reverted_cropping = np.zeros((predicted_probabilities.shape[0], *original_shape),
                                           dtype=predicted_probabilities.dtype) \
            if isinstance(predicted_probabilities, np.ndarray) else \
            torch.zeros((predicted_probabilities.shape[0], *original_shape), dtype=predicted_probabilities.dtype)

        probs_reverted_cropping[0] = 1

        slicer = bounding_box_to_slice(bbox)
        probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_probabilities
        return probs_reverted_cropping

    @staticmethod
    def filter_background(classes_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]]):
        # heck yeah
        # This is definitely taking list comprehension too far. Enjoy.
        return [i for i in classes_or_regions if
                ((not isinstance(i, (tuple, list))) and i != 0)
                or
                (isinstance(i, (tuple, list)) and not (
                        len(np.unique(i)) == 1 and np.unique(i)[0] == 0))]


    @property
    def foreground_labels(self):
        return self.filter_background(self.all_labels)

    @property
    def num_output_heads(self):

        return len(self.all_labels)


def get_labelmanager_class_from_plans(plans: dict) -> Type[LabelManager]:
    if 'label_manager' not in plans.keys():
        print('No label manager specified in plans. Using default: LabelManager')
        return LabelManager
    else:
        labelmanager_class = recursive_find_python_class(join(nndosenet.__path__[0], "utilities", "label_handling"),
                                                         plans['label_manager'],
                                                         current_module="nndosenet.utilities.label_handling")
        return labelmanager_class




def determine_num_input_channels(plans_manager: PlansManager,
                                 configuration_or_config_manager: Union[str, ConfigurationManager],
                                 dataset_json: dict) -> int:

    num_modalities = len(dataset_json['modality']) if 'modality' in dataset_json.keys() else len(dataset_json['channel_names'])
    
    return num_modalities




    
    
