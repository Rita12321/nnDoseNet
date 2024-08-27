# System requirements

## Operating System
nnDoseNet has been tested on Linux (Ubuntu 22.04.4 LTS)
## Hardware requirements
We support GPU (tested), as for CPU and Apple M1/M2 as devices (currently Apple mps does not implement 3D 
convolutions, so you might have to use the CPU on those devices) theoretically would work but not tested at all ( I wouldn't mind testing for you if you send me some free Apple device for testing ; )  

### Hardware requirements 
We trained nnDoseNet on NVIDIA Corporation TU102 [TITAN RTX] on lab PC and A100, P100 from our institutional HPC center.

For training a GPU with at least 10 GB (popular non-data center options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is 
required. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads) 
are the bare minimum! CPU requirements are mostly related to data augmentation and scale with the number of inputs. Plus, the faster the GPU, the better the CPU should be!


### Setting the correct number of Workers for data augmentation (training only)
Note that you will need to manually set the number of processes nnDoseNet uses for data augmentation according to your 
CPU/GPU ratio. You can set nps for training by 
setting the `nnDose_n_proc_DA` environment variable (`export nnUNet_n_proc_DA=XX`). 
Recommended values (assuming a recent CPU with good IPC) are 10-12 for RTX 2080 ti, 12 for a RTX 3090, 16-18 for 
RTX 4090, 28-32 for A100. Optimal values may vary depending on the number of input channels/modalities and number of classes.

# Installation instructions
We strongly recommend that you install nnDoseNet in a virtual environment! Pip or anaconda are both fine. If you choose to 
compile PyTorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!


1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum speed, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install nnDoseNet depending on your use case:
  
    This will create a copy of the nnU-Net code on your computer so that you
   can modify it as needed:

          ```bash
          git clone https://github.com/Rita12321/nnDoseNet
          cd nnDoseNet
          pip install -e .
          ```
3) nnDoseNet needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few environment variables. Please follow the instructions [here](setting_up_paths.md).

Installing nnDoseNet will add several new commands to your terminal. These commands are used to run the entire nnDoseNet
pipeline. You can execute them from any location on your system. All nnDoseNet commands have the prefix `nnDoseNet_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnDoseNet in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the entry_points in the setup.py file.

All nnDoseNet commands have a `-h` option which gives information on how to use them. \
*It would be much more clear to check the input arguments of commands with `-h` .\
*Most inference command has `_Ts` version for easier inference pipeline for test dataset in the experiments.