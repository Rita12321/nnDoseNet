#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""
nnDose_dicom = os.environ.get('nnDose_dicom')
nnDose_raw = os.environ.get('nnDose_raw')
nnDose_preprocessed = os.environ.get('nnDose_preprocessed')
nnDose_results = os.environ.get('nnDose_results')
nnDose_prediction = os.environ.get('nnDose_prediction')


if nnDose_raw is None:
    print("nnDose_raw is not defined and nnDoseNet can only be used on data for which preprocessed files "
          "are already present on your system. nnDoseNet cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnDose_preprocessed is None:
    print("nnDose_preprocessed is not defined and nnDoseNet can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnDose_results is None:
    print("nnDose_results is not defined and nnDoseNet cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
if nnDose_dicom is None:
    print("nnDose_dicom is not defined and if you want to use the dicom_converter or prelabel function, this folder would be neccessary")
    