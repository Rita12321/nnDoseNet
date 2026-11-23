import numpy as np
from nndosenet.training.dataloading.base_data_loader import nnUNetDataLoaderBase

class nnDoseDataLoader3D(nnUNetDataLoaderBase):

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        gt_all = np.zeros(self.gt_shape, dtype=np.float32)#NOTE:changed np.int16-->np.float32
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, gt, properties = self._data.load_case(i)



            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, gt.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            gt = gt[this_slice]



            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)#0
            gt_all[j] = np.pad(gt, ((0, 0), *padding), 'constant', constant_values=-1)#



        return {'data': data_all, 'seg': gt_all, 'properties': case_properties, 'keys': selected_keys}

class CycleGANDataLoader3D(nnUNetDataLoaderBase):
    """
    DataLoader for CycleGAN, adapted from nnDoseDataLoader3D.
    Loads random 3D patches of image data from a single domain.
    Intended to be used in pairs (one loader for domain A, one for domain B)
    by the CycleGANTrainer.
    """
    def __init__(self, dataset, batch_size, patch_size, final_patch_size, label_manager,
                 oversample_foreground_percent=0.0, # Default to 0 for CycleGAN
                 sampling_probabilities=None, pad_sides=None, force_random_patch=True):
        """
        Initializes the DataLoader.

        Args:
            dataset: The dataset object (e.g., nnDoseDataset) containing data for ONE domain.
            batch_size: Number of samples per batch.
            patch_size: The size of the patch to sample (can be different from final_patch_size if using initial_patch_size logic).
            final_patch_size: The final patch size after potential augmentation/resizing (used to determine padding).
            label_manager: Label manager (might be less relevant if not using segmentation-based sampling).
            oversample_foreground_percent: Probability of sampling foreground (set to 0.0 for random patches).
            sampling_probabilities: Optional sampling probabilities for cases.
            pad_sides: Optional padding configuration.
            force_random_patch (bool): If True, ignores oversampling and always selects a random patch location.
                                       Recommended for standard CycleGAN.
        """
        super().__init__(dataset, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides)
        # Ensure the correct shapes are set based on final_patch_size
        # Assuming num_channels is derived correctly in base class init for data
        self.data_shape = (batch_size, self.num_channels_data, *self.patch_size)
        # We don't need a separate gt_shape if we are not loading/processing gt
        self.force_random_patch = force_random_patch
        self.print_to_log(f"CycleGANDataLoader3D initialized. force_random_patch={self.force_random_patch}, oversample={self.oversample_foreground_percent}")


    def generate_train_batch(self):
        """
        Generates a batch of random image patches from the assigned domain.
        """
        selected_keys = self.get_indices()
        # preallocate memory for data
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        # No need for gt_all in standard CycleGAN
        case_properties_all = [] # Store properties if needed downstream

        for j, case_id in enumerate(selected_keys):

            properties = self._data.get_properties(case_id) # Get properties first
            # Load only data (channel 0) - assuming GT/seg is not needed here
            # Adapt this if your 'data' includes segmentation channels you need to ignore
            data = self._data.load_data(case_id, 0) # Load only the image data

            shape = data.shape[1:] # Shape without channel dim
            dim = len(shape)

            if self.force_random_patch:
                 # --- Simplified Random Patch Selection ---
                 # Ensure patch fits within shape after potential padding
                 need_padding = [ps > s for ps, s in zip(self.patch_size, shape)]
                 if any(need_padding):
                      # Calculate necessary padding
                      padding_before = []
                      padding_after = []
                      shape_padded = list(shape)
                      for d in range(dim):
                           pad_total = max(0, self.patch_size[d] - shape[d])
                           pad_b = pad_total // 2
                           pad_a = pad_total - pad_b
                           padding_before.append(pad_b)
                           padding_after.append(pad_a)
                           shape_padded[d] += pad_total
                      # Pad the data first
                      data = np.pad(data, ((0, 0), *zip(padding_before, padding_after)), 'constant', constant_values=0)
                      shape = data.shape[1:] # Update shape

                 # Now select a random valid start coordinate
                 lbs = [np.random.randint(0, s - ps + 1) for s, ps in zip(shape, self.patch_size)]
                 ubs = [lb + ps for lb, ps in zip(lbs, self.patch_size)]
                 # Slice the (potentially padded) data
                 this_slice = tuple([slice(0, data.shape[0])] + [slice(lb, ub) for lb, ub in zip(lbs, ubs)])
                 data_patch = data[this_slice]
                 data_all[j] = data_patch
                 # --- End Simplified Patch Selection ---

            else:
                 # --- Original BBox Logic (if needed, e.g., based on content) ---
                 force_fg = self.get_do_oversample(j)
                 # get_bbox might need class_locations from properties, ensure properties are loaded
                 bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties.get('class_locations', None)) # Use .get for safety

                 valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
                 valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

                 this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
                 data_cropped = data[this_slice]

                 # Calculate padding needed to reach self.patch_size
                 padding = [(ul - ll, ur - lr) for ll, ur, ps, sl in zip(valid_bbox_lbs, valid_bbox_ubs, self.patch_size, shape)]
                 padding_dims = []
                 shape_after_crop = data_cropped.shape[1:]
                 for d in range(dim):
                      pad_total_dim = self.patch_size[d] - shape_after_crop[d]
                      assert pad_total_dim >= 0, "Cropped shape larger than patch size?"
                      pad_before = -min(0, bbox_lbs[d]) # Padding added if bbox started outside image
                      pad_after = max(bbox_ubs[d] - shape[d], 0) # Padding added if bbox ended outside image
                      # We need pad_total_dim = pad_before + pad_after
                      # This logic seems complex, let's recalculate padding directly
                      pad_needed = [(self.patch_size[d] - shape_after_crop[d]) for d in range(dim)]
                      pad_before_axis = [p // 2 for p in pad_needed]
                      pad_after_axis = [p - pb for p, pb in zip(pad_needed, pad_before_axis)]
                      padding_final = list(zip(pad_before_axis, pad_after_axis))

                 # Pad the cropped data to final patch size
                 data_all[j] = np.pad(data_cropped, ((0, 0), *padding_final), 'constant', constant_values=0)
                 # --- End Original BBox Logic ---


            case_properties_all.append(properties)


        # Return only data, properties, and keys
        # The transforms later will expect 'data' and potentially 'target' if renamed
        # Let's return 'data' for now. The NumpyToTensor will handle it.
        return {'data': data_all, 'properties': case_properties_all, 'keys': selected_keys}

    def print_to_log(self, *args):
        """ Simple print wrapper """
        # Avoid crashing if run standalone where parent logger might not exist
        print("CycleGANDataLoader3D:", *args)
