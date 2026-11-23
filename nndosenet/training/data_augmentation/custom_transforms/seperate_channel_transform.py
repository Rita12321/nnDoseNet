from typing import Tuple, Union, List

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class SeperateDataMask(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self,mask_channel:Union[List, Tuple], input_key: str = "data",
                 output_key: str = "data_mask"):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """

        self.output_key = output_key
        self.input_key = input_key
        self.mask_channel=mask_channel

    def __call__(self, **data_dict):


        data_channel=list(set(np.arange(data_dict[self.input_key].shape[1])) - set(self.mask_channel))
        data_shape=data_dict[self.input_key].shape
        data_dict[self.output_key] =data_dict[self.input_key][:,self.mask_channel,...]
        data_dict[self.input_key] =data_dict[self.input_key][:,data_channel,...]


        return data_dict


class CopyDataMask(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self,mask_channel:Union[List, Tuple], input_key: str = "data",
                 output_key: str = "data_mask"):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """

        self.output_key = output_key
        self.input_key = input_key
        self.mask_channel=mask_channel

    def __call__(self, **data_dict):
        data_dict[self.output_key] =data_dict[self.input_key][:,self.mask_channel,...]


        return data_dict


class revert_SeperateDataMask(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self,mask_channel:Union[List, Tuple], input_key: str = "data",
                 output_key: str = "data_mask",dsv=False):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.dsv=dsv
        self.output_key = output_key
        self.input_key = input_key
        self.mask_channel=mask_channel

    def __call__(self, **data_dict):
        if self.dsv:
            mask=data_dict[self.output_key][0]
        else:
            mask=data_dict[self.output_key]


        all_channel=list(set(np.arange(int(data_dict[self.input_key].shape[1]+mask.shape[1]))))
        data_channel=list(set(all_channel)- set(self.mask_channel))
        index_data=0
        index_mask=0

        if int(0) in data_channel:
            out= np.expand_dims(data_dict[self.input_key][:,0,...], axis=1)
            index_data=index_data+1
        else:
            out=np.expand_dims(mask[:,0,...], axis=1)
            index_mask=index_mask+1
        for c in all_channel:
            if c ==0:
                continue
            if c in data_channel:
                out=np.concatenate((out,np.expand_dims(data_dict[self.input_key][:,index_data,...], axis=1)),axis=1)
                index_data=index_data+1
            else:
                out=np.concatenate((out,np.expand_dims(mask[:,index_mask,...], axis=1)),axis=1)
                index_mask=index_mask+1

        data_dict[self.input_key]=out

        return data_dict
