# Intensity normalization of different modality (channel)
>Please read `dataset.json` section in [dataset_format(nnDose_raw)](dataset_format(nnDose_raw).md) first.

The type of intensity normalization applied in nnDoseNet can be controlled via the `channel_names` 
entry in the dataset.json.Channel name represent the modality of it and the correspond normalization method on each channel.\
The `channel_names` typically looks like this in nnDoseNet: 

        "channel_names": {
            "0": "CT",
            "1": "Target",
            "2": "OAR",
            "3": "BODY"
        },

Here is the avaliable normalization method and corrrespond names.
If you enter a channel name that is not in the following list, the default (`zscore`) will be applied.


- `CT`: Perform CT normalization. Specifically, collect intensity values from the foreground (where intensity in dose map is <=0) in all training cases, compute the mean, standard deviation as well as the 0.5 and 
99.5 percentile of the values. Then clip to the percentiles, followed by subtraction of the mean and division with the 
standard deviation. The normalization that is applied is the same for each training case (for this input channel).
The values used by nnU-Net for normalization are stored in the `foreground_intensity_properties_per_channel` entry in the 
corresponding plans file. This normalization is suitable for modalities presenting physical quantities such as CT 
images and ADC maps.
- `PTV`, `Target`, `OAR`, `Body`, `mask`: do not perform any normalization at all and set dtype to uint8
- `noNorm` : do not perform any normalization at all
- `rescale_to_0_1`: rescale the intensities to [0, 1]
- `rgb_to_0_1`: assumes uint8 inputs. Divides by 255 to rescale uint8 to [0, 1]

- `zscore`/anything else: perform z-scoring (subtract mean and standard deviation) separately for each train case


