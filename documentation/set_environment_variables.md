# How to set environment variables

nnDoseNet requires some environment variables so that it always knows where the dicom(unlabel) data, raw data, preprocessed data trained models and prediction data are. Depending on the operating system, these environment variables need to be set in different ways.

Variables can either be set permanently (recommended!) or you can decide to set them everytime you call nnDoseNet. 

# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export nnDose_dicom="/home/changh/Documents/nnDose_1_dicom"
export nnDose_raw="/home/changh/Documents/nnDose_2_raw"
export nnDose_preprocessed="/home/changh/Documents/nnDose_3_preprocess"
export nnDose_results="/home/changh/Documents/nnDose_4_result"
export nnDose_prediction="/home/changh/Documents/nnDose_5_prediction"
```

(of course you need to adapt the paths to the actual folders you intend to use).
If you are using a different shell, such as zsh, you will need to find the correct script for it. For zsh this is `.zshrc`.

## Temporary
Just execute the following lines whenever you run nnU-Net:
```bash
export nnDose_dicom="/home/changh/Documents/nnDose_1_dicom"
export nnDose_raw="/home/changh/Documents/nnDose_2_raw"
export nnDose_preprocessed="/home/changh/Documents/nnDose_3_preprocess"
export nnDose_results="/home/changh/Documents/nnDose_4_result"
export nnDose_prediction="/home/changh/Documents/nnDose_5_prediction"
```
(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your terminal! They will also only apply to the current 
terminal window and DO NOT transfer to other terminals!

Alternatively you can also just prefix them to your nnU-Net commands:

`nnDose_dicom="/home/changh/Documents/nnDose_1_dicom" nnDose_raw="/home/changh/Documents/nnDose_2_raw" nnDoseNet_train[...]`

## Verify that environment parameters are set
You can always execute `echo ${nnDose_raw}` etc to print the environment variables. This will return an empty string if 
they were not set.

# Windows (Not tested)
Useful links:
- [https://www3.ntu.edu.sg](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=To%20set%20(or%20change)%20a,it%20to%20an%20empty%20string.)
- [https://phoenixnap.com](https://phoenixnap.com/kb/windows-set-environment-variable)

## Permanent
See `Set Environment Variable in Windows via GUI` [here](https://phoenixnap.com/kb/windows-set-environment-variable). 
Or read about setx (command prompt).

## Temporary
Just execute the following before you run nnDoseNet:

(powershell)
```powershell
$Env:nnDose_dicom="/home/changh/Documents/nnDose_1_dicom"
$Env:nnDose_raw="/home/changh/Documents/nnDose_2_raw"
$Env:nnDose_preprocessed="/home/changh/Documents/nnDose_3_preprocess"
$Env:nnDose_results="/home/changh/Documents/nnDose_4_result"
$Env:nnDose_prediction="/home/changh/Documents/nnDose_5_prediction"
```

(command prompt)
```commandline
set nnDose_dicom="/home/changh/Documents/nnDose_1_dicom"
set nnDose_raw="/home/changh/Documents/nnDose_2_raw"
set nnDose_preprocessed="/home/changh/Documents/nnDose_3_preprocess"
set nnDose_results="/home/changh/Documents/nnDose_4_result"
set nnDose_prediction="/home/changh/Documents/nnDose_5_prediction"
```

(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your session! They will also only apply to the current 
window and DO NOT transfer to other sessions!

## Verify that environment parameters are set
Printing in Windows works differently depending on the environment you are in:

powershell: `echo $Env:[variable_name]`

command prompt: `echo %[variable_name]%`