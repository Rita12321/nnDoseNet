from typing import Tuple, Union, List

from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        if self.axes is None:
            axes = list(range(2, len(data_dict[self.input_key].shape)))
        else:
            axes = self.axes

        output = []
        if self.output_key is 'ddp_data':
            
            
            for s in self.ds_scales:
                if not isinstance(s, (tuple, list)):
                    s = [s] * len(axes)
                else:
                    assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                                f'for each axis) then the number of entried in that tuple (here ' \
                                                f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

                data=data_dict[self.input_key][:,1:,...]
                data=data.astype(int)         
                if all([i == 1 for i in s]):#if[1,1,1]
                    output.append(data)
                else:
                    new_shape = np.array(data.shape).astype(int)
                    for i, a in enumerate(axes):
                        new_shape[a] *= s[i]
                    new_shape = np.round(new_shape).astype(int)
                    out_seg = np.zeros(new_shape, dtype=int)
                    for b in range(data.shape[0]):
                        for c in range(1,data.shape[1]):
                            out_seg[b, c] = resize_segmentation(data[b, c], new_shape[2:],order=0)#self.order
                    output.append(out_seg)
                    # print("data 1",np.unique(data[:,0,...]))
                    # # print("data 2",np.unique(data[:,1,...]))
                    # print("out  1",np.unique(out_seg[:,0,...]))
                    #print("out  2",np.unique(out_seg[:,1,...]))
            # for i in range(0,5):
            #     print()
        else:
            
            #print("input_shape_ddp:",data_dict[self.input_key].shape," key:",self.input_key,"   axes:",axes,"   ds_scales",self.ds_scales)
            for s in self.ds_scales:
                if not isinstance(s, (tuple, list)):
                    s = [s] * len(axes)
                else:
                    assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                                f'for each axis) then the number of entried in that tuple (here ' \
                                                f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

                if all([i == 1 for i in s]):#if[1,1,1]
                    output.append(data_dict[self.input_key])
                else:
                    new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                    for i, a in enumerate(axes):
                        new_shape[a] *= s[i]
                    new_shape = np.round(new_shape).astype(int)
                    out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                    for b in range(data_dict[self.input_key].shape[0]):
                        for c in range(data_dict[self.input_key].shape[1]):
                            out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)#self.order
                    output.append(out_seg)
        #print(self.output_key,len(output))
        data_dict[self.output_key] = output
        return data_dict
