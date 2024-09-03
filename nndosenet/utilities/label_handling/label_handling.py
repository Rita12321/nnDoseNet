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
        # self._has_regions = False
        
        # self._ignore_label: Union[None, int] = None #TODO: self._determine_ignore_label()
        self._all_labels: List[int] = self._get_all_labels()

        # self._regions: Union[None, List[Union[int, Tuple[int, ...]]]] = self._get_regions()

        # if self.has_ignore_label:
            
        #     assert self.ignore_label == max(
        #         self.all_labels) + 1, 'If you use the ignore label it must have the highest ' \
        #                               'label value! It cannot be 0 or in between other labels. ' \
        #                               'Sorry bro.'


        # self.inference_nonlin = None #inference_nonlin

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

    # def _get_regions(self) -> Union[None, List[Union[int, Tuple[int, ...]]]]: # TODO: just keeping this for futurework
    #     # if not self._has_regions or self._force_use_labels:
    #     #     return None
    #     # else:
    #     #     assert self.regions_class_order is not None, 'if region-based training is requested then you need to ' \
    #     #                                                  'define regions_class_order!'
    #     #     regions = []
    #     #     for k, r in self.label_dict.items():
    #     #         # ignore ignore label
    #     #         if k == 'ignore':
    #     #             continue
    #     #         # ignore regions that are background
    #     #         if (np.isscalar(r) and r == 0) \
    #     #                 or \
    #     #                 (isinstance(r, (tuple, list)) and len(np.unique(r)) == 1 and np.unique(r)[0] == 0):
    #     #             continue
    #     #         if isinstance(r, list):
    #     #             r = tuple(r)
    #     #         regions.append(r)
    #     #     assert len(self.regions_class_order) == len(regions), 'regions_class_order must have as ' \
    #     #                                                           'many entries as there are ' \
    #     #                                                           'regions'
    #     #     return regions
    #     return None

    # def _determine_ignore_label(self) -> Union[None, int]: # TODO: just keeping this for futurework
    #     pass
    #     # ignore_label = self.label_dict.get('ignore')
    #     # if ignore_label is not None:
    #     #     assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
    #     #                                           f'(list/tuple). Got {type(ignore_label)}.'
    #     # return ignore_label
    #     return None

    # @property
    # def has_regions(self) -> bool:
    #     return self._has_regions

    # @property
    # def has_ignore_label(self) -> bool:
    #     return self.ignore_label is not None

    # @property
    # def all_regions(self) -> Union[None, List[Union[int, Tuple[int, ...]]]]:
    #     return self._regions

    @property
    def all_labels(self) -> List[int]:
        return self._all_labels

    # @property
    # def ignore_label(self) -> Union[None, int]:
    #     return self._ignore_label

    # def apply_inference_nonlin(self, logits: Union[np.ndarray, torch.Tensor]) -> \
    #         Union[np.ndarray, torch.Tensor]:
    #     """
    #     logits has to have shape (c, x, y(, z)) where c is the number of classes/regions
    #     """
    #     if isinstance(logits, np.ndarray):
    #         logits = torch.from_numpy(logits)

    #     with torch.no_grad():
    #         probabilities = logits.float()
    #         # logits = logits.float()
    #         # probabilities = self.inference_nonlin(logits)

    #     return probabilities

    def probability_squeeze_channel(self, predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:

        
        if isinstance(predicted_probabilities,  torch.Tensor):
            predicted_probabilities.cpu().numpy()
        if predicted_probabilities.dtype != np.float32:
            predicted_probabilities = predicted_probabilities.astype(np.float32)
            
        # predicted_probabilities = predicted_probabilities.argmax(0)
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

        # if not self.has_regions:
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

    # @property
    # def foreground_regions(self):
    #     return self.filter_background(self.all_regions)

    @property
    def foreground_labels(self):
        return self.filter_background(self.all_labels)

    @property
    def num_output_heads(self):
        # if self.has_regions:
        #     return len(self.foreground_regions)
        # else:
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


# def convert_labelmap_to_one_hot(segmentation: Union[np.ndarray, torch.Tensor],
#                                 all_labels: Union[List, torch.Tensor, np.ndarray, tuple],
#                                 output_dtype=None) -> Union[np.ndarray, torch.Tensor]:
#     #It's not really going to convert anything, just changing shape since in nndosenet len(all_labels) is always 1
#     #just keepimg the function for future use
#     """
#     if output_dtype is None then we use np.float32/torch.float32
#     if input is torch.Tensor then output will be on the same device

#     np.ndarray is faster than torch.Tensor

#     if segmentation is torch.Tensor, this function will be faster if it is LongTensor. If it is somethine else we have
#     to cast which takes time.

#     IMPORTANT: This function only works properly if your labels are consecutive integers, so something like 0, 1, 2, 3, ...
#     DO NOT use it with 0, 32, 123, 255, ... or whatever (fix your labels, yo)
#     """
#     print("oh noooooooooooooooooooooooooooooooooooooooooooooooooooooo")
#     if isinstance(segmentation, torch.Tensor):
#         result = torch.zeros((len(all_labels), *segmentation.shape),
#                              dtype=output_dtype if output_dtype is not None else torch.float32,
#                              device=segmentation.device)
#         # variant 1, 2x faster than 2
#         result.scatter_(0, segmentation[None].long(), 1)  # why does this have to be long!?
#         # variant 2, slower than 1
#         # for i, l in enumerate(all_labels):
#         #     result[i] = segmentation == l
#     else:
#         result = np.zeros((len(all_labels), *segmentation.shape),
#                           dtype=output_dtype if output_dtype is not None else np.float32)
#         # variant 1, fastest in my testing
#         for i, l in enumerate(all_labels):
#             result[i] = segmentation == l
#         # variant 2. Takes about twice as long so nah
#         # result = np.eye(len(all_labels))[segmentation].transpose((3, 0, 1, 2))
#     return result


def determine_num_input_channels(plans_manager: PlansManager,
                                 configuration_or_config_manager: Union[str, ConfigurationManager],
                                 dataset_json: dict) -> int:
    # if isinstance(configuration_or_config_manager, str):
    #     config_manager = plans_manager.get_configuration(configuration_or_config_manager)
    # else:
    #     config_manager = configuration_or_config_manager

    # label_manager = plans_manager.get_label_manager(dataset_json)
    num_modalities = len(dataset_json['modality']) if 'modality' in dataset_json.keys() else len(dataset_json['channel_names'])
    
    return num_modalities
    # # cascade has different number of input channels
    # if config_manager.previous_stage_name is not None:
    #     num_label_inputs = len(label_manager.foreground_labels)
    #     num_input_channels = num_modalities + num_label_inputs
    # else:
    #     num_input_channels = num_modalities
    # return num_input_channels



    
    
