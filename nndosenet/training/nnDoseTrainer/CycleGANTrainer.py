import itertools # Add this import for cycling through dataloaders
from nndosenet.training.nnDoseTrainer.nnDoseTrainer import nnDoseTrainer
import torch
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from nndosenet.configuration import ANISO_THRESHOLD, default_num_processes
from nndosenet.evaluation.evaluate_predictions import compute_dose_metrics_on_folder
from nndosenet.inference.export_prediction import export_prediction_from_logits
from nndosenet.inference.predict_from_raw_data import nnDoseNetPredictor
from nndosenet.inference.sliding_window_prediction import compute_gaussian
from nndosenet.paths import nnDose_preprocessed, nnDose_results, nnDose_raw
from nndosenet.training.data_augmentation.compute_initial_patch_size import get_patch_size

from nndosenet.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nndosenet.training.data_augmentation.custom_transforms.seperate_channel_transform import SeperateDataMask, revert_SeperateDataMask, CopyDataMask
from nndosenet.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nndosenet.training.data_augmentation.custom_transforms.masking import MaskTransform
# from nndosenet.training.data_augmentation.custom_transforms.region_based_training import \
#     ConvertSegmentationToRegionsTransform
from nndosenet.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform

from nndosenet.training.dataloading.data_loader_3d import CycleGANDataLoader3D
from nndosenet.training.dataloading.nndose_dataset import nnDoseDataset
from nndosenet.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nndosenet.training.logging.nnunet_logger import nnUNetLogger
from nndosenet.training.loss.compound_losses import Err_and_DVH_Loss
from nndosenet.training.loss.deep_supervision import DeepSupervisionWrapper

from nndosenet.training.loss.DVH_loss import value_DVH_Loss, criteria_DVH_Loss
from nndosenet.training.lr_scheduler.polylr import PolyLRScheduler
from nndosenet.utilities.collate_outputs import collate_outputs
from nndosenet.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nndosenet.utilities.file_path_utilities import check_workers_alive_and_busy
from nndosenet.utilities.get_network_from_plans import get_network_from_plans
from nndosenet.utilities.helpers import empty_cache
from nndosenet.utilities.label_handling.label_handling import determine_num_input_channels
from nndosenet.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nndosenet.training.nnDoseTrainer.nnDoseTrainer import nnDoseTrainer
from sklearn.model_selection import KFold
from torch import nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
# Assuming nnDoseTrainer is defined in the same file or imported
# from nndosenet.training.nnDoseTrainer.nnDoseTrainer import nnDoseTrainer

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nndosenet.utilities.cycleGAN import CycleGANGenerator, PatchGANDiscriminator



class CycleGANTrainer(nnDoseTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 #dataset_json_domain_b: dict=None, # Add dataset_json for the second domain
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """
        Trainer for CycleGAN. Inherits from nnDoseTrainer for setup, logging, etc.
        but overrides network, optimizer, loss, and training/validation steps.

        Args:
            plans (dict): nnU-Net plans dictionary (used for generator architecture).
            configuration (str): Configuration name from plans.
            fold (int): Cross-validation fold.
            dataset_json (dict): Dataset JSON for Domain A.
            dataset_json_domain_b (dict): Dataset JSON for Domain B. Needed to determine
                                           input/output channels if they differ.
            unpack_dataset (bool, optional): Whether to unpack data. Defaults to True.
            device (torch.device, optional): Device to use. Defaults to torch.device('cuda').
        """
        # Call parent init FIRST to set up paths, logging, device, ddp etc.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # --- CycleGAN Specific Hyperparameters ---
        self.lambda_cycle = 10.0  # Cycle loss weight
        self.lambda_identity = 0.5 * self.lambda_cycle # Identity loss weight (often 0 or 5.0)
        self.lr_g = 0.0002  # Learning rate for generators
        self.lr_d = 0.0002  # Learning rate for discriminators
        self.beta1 = 0.5     # Adam beta1
        # if dataset_json_domain_b is None:
        #      self.dataset_json_domain_b = dataset_json_domain_b

        # else:
        self.dataset_json_domain_b = dataset_json
       #dataset_json_domain_b # Store domain B info

        # --- Override/Reset Placeholders from Parent ---
        # We manage multiple networks and optimizers
        self.network = None # Parent's single network placeholder not used directly
        self.optimizer = None # Parent's single optimizer placeholder not used
        self.lr_scheduler = None # Use CycleGAN specific LR scheduling if needed (often linear decay)
        self.loss = None # Parent's loss (_build_loss) not used directly

        # CycleGAN components - Initialized in self.initialize()
        self.G_AtoB = None
        self.G_BtoA = None
        self.D_A = None
        self.D_B = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.criterion_GAN = None
        self.criterion_Cycle = None
        self.criterion_Identity = None

        # Placeholder for dataloaders for domain A and B
        self.dataloader_A = None
        self.dataloader_B = None

        # Adjust logging if needed (CycleGAN losses are different)
        self.logger = nnUNetLogger(verbose=False) # Re-init logger if you want different keys

        # Modify output folder name slightly maybe? (Optional)
        # self.output_folder_base = join(...) # Can adjust the name if desired
        # self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        # self.log_file = join(...) # Re-point log file if folder changed

        # Make sure was_initialized is False so parent doesn't complain
        self.was_initialized = False


    def initialize(self):
        """
        Overrides parent initialize. Sets up the 4 CycleGAN networks, 2 optimizers,
        and loss criteria.
        """
        if not self.was_initialized:
            # Determine Generator Network Params from Plans
            # Assuming Domain A -> Domain B mapping primarily drives the plans
            # Use Domain A's dataset_json for input channels
            # Use Domain B's dataset_json for output channels (if different)
            num_input_channels_A = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
            num_input_channels_B = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json_domain_b)
            # Output channels of G_AtoB should match input channels of Domain B images
            # Output channels of G_BtoA should match input channels of Domain A images

            self.print_to_log_file(f"Generator Input Channels (Domain A): {num_input_channels_A}")
            self.print_to_log_file(f"Generator Output Channels (Domain B): {num_input_channels_B}")

            # --- Build Generators using CycleGANGenerator ---
            # Note: We pass num_input/output channels explicitly
            # The build_network_architecture static method is not directly used here,
            # as we need specific Generator class.
            num_stages = len(self.configuration_manager.conv_kernel_sizes)

            dim = len(self.configuration_manager.conv_kernel_sizes[0])
            conv_op = convert_dim_to_conv_op(dim)
            block_type = self.configuration_manager.block if hasattr(self.configuration_manager, 'block') else BasicBlockD

            # Get bottleneck channels - default should be None if using BasicBlockD
            bottleneck_channels_config = getattr(self.configuration_manager, 'network_arch_bottleneck_channels', None)

            # Get stem channels - default is None, letting ResidualEncoder handle it (often uses features_per_stage[0])
            stem_channels_config = getattr(self.configuration_manager, 'network_arch_stem_channels', None)
            common_kwargs = dict(
                 n_stages = num_stages,
                         features_per_stage=[min(self.configuration_manager.UNet_base_num_features * 2 ** i,
                                self.configuration_manager.unet_max_num_features) for i in range(num_stages)],
                 conv_op=conv_op,
                 kernel_sizes=self.configuration_manager.conv_kernel_sizes,
                 strides = self.configuration_manager.pool_op_kernel_sizes,
                 n_blocks_per_stage = self.configuration_manager.n_conv_per_stage_encoder,
                 n_conv_per_stage_decoder = self.configuration_manager.n_conv_per_stage_decoder,
                 conv_bias = True,
                 norm_op = get_matching_instancenorm(conv_op), # Use norm from plans
                 norm_op_kwargs = {'eps': 1e-5, 'affine': True},
                 dropout_op = None,
                 dropout_op_kwargs = None,
                 nonlin = nn.LeakyReLU,
                 nonlin_kwargs = {'inplace': True},
                 block = block_type, # Likely BasicBlockD
                 bottleneck_channels = bottleneck_channels_config, # Likely None
                 stem_channels = stem_channels_config  # Likely None (effective default ~128)
            )
            self.G_AtoB = CycleGANGenerator(num_input_channels_A, num_input_channels_B, **common_kwargs).to(self.device)
            self.G_BtoA = CycleGANGenerator(num_input_channels_B, num_input_channels_A, **common_kwargs).to(self.device)

            # --- Build Discriminators ---
            # Use norm_op from plans for consistency, or default to InstanceNorm
            disc_norm_op = common_kwargs['norm_op'] if common_kwargs['norm_op'] is not None else nn.InstanceNorm2d # Assuming 2D default if needed
            # Adapt 2d/3d based on conv_op
            if common_kwargs['conv_op'] == nn.Conv3d:
                 disc_norm_op = nn.InstanceNorm3d if common_kwargs['norm_op'] is None else common_kwargs['norm_op']

            self.D_A = PatchGANDiscriminator(num_input_channels_A, norm_op=disc_norm_op).to(self.device)
            self.D_B = PatchGANDiscriminator(num_input_channels_B, norm_op=disc_norm_op).to(self.device)

            # --- Initialize Weights (Optional but often done for GANs) ---
            # You might add weight initialization here (e.g., normal, xavier)
            # init_weights(self.G_AtoB, init_type='normal', init_gain=0.02)
            # init_weights(self.G_BtoA, init_type='normal', init_gain=0.02)
            # init_weights(self.D_A, init_type='normal', init_gain=0.02)
            # init_weights(self.D_B, init_type='normal', init_gain=0.02)
            # print("Network weights initialized.")


            # --- Configure Optimizers ---
            self.optimizer_G, self.optimizer_D = self.configure_optimizers() # Call the overridden method

            # --- Build Loss Functions ---
            # No need to call parent's _build_loss
            self.criterion_GAN = torch.nn.MSELoss()  # Use MSELoss for LSGAN stability
            self.criterion_Cycle = torch.nn.L1Loss()
            self.criterion_Identity = torch.nn.L1Loss()

            # --- DDP Wrapping (Apply to all relevant networks) ---
            if self.is_dsv:
                self.print_to_log_file("Wrapping models in DDP...")
                # Convert BatchNorm before wrapping if using BatchNorm
                # self.G_AtoB = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_AtoB)
                # self.G_BtoA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_BtoA)
                # self.D_A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.D_A)
                # self.D_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.D_B)

                self.G_AtoB = DDP(self.G_AtoB, device_ids=[self.local_rank], find_unused_parameters=True) # find_unused might be needed
                self.G_BtoA = DDP(self.G_BtoA, device_ids=[self.local_rank], find_unused_parameters=True)
                self.D_A = DDP(self.D_A, device_ids=[self.local_rank], find_unused_parameters=True)
                self.D_B = DDP(self.D_B, device_ids=[self.local_rank], find_unused_parameters=True)
                self.print_to_log_file("DDP wrapping done.")

            self.was_initialized = True
        else:
            # This should not happen with the new structure
             self.print_to_log_file("Warning: initialize called again when already initialized.")
            # raise RuntimeError("Initialize called again")

    def configure_optimizers(self):
        """ Overrides parent. Configures optimizers for Generators and Discriminators."""
        self.print_to_log_file("Configuring Adam optimizers for CycleGAN...")
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()),
            lr=self.lr_g, betas=(self.beta1, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.lr_d, betas=(self.beta1, 0.999)
        )
        # CycleGAN often uses a linear decay scheduler, not PolyLR
        # We can implement this in the training loop or via LambdaLR if needed.
        # For now, return optimizers; handle scheduling in the loop if required.
        self.lr_scheduler = None # Explicitly disable parent's scheduler handling
        return optimizer_G, optimizer_D

    def _build_loss(self):
        """ Overrides parent. Losses are defined in initialize for CycleGAN. """
        self.print_to_log_file("CycleGAN losses defined during initialization.")
        return None # Return None to satisfy parent class if it checks

    def get_dataloaders(self):
        """
        Overrides parent. Must return dataloaders for Domain A and Domain B.
        Assumes dataset structure allows differentiating domains (e.g., by subfolders
        or file naming conventions identifiable by nnDoseDataset or a custom one).

        This implementation is a placeholder and likely needs significant adaptation
        based on how your Domain A and Domain B data are stored and preprocessed.
        """
        self.print_to_log_file("Getting CycleGAN Dataloaders (Placeholder - Requires Adaptation)...")

        # --- Configuration ---
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # Deep supervision scales not needed for CycleGAN target processing
        # deep_supervision_scales = self._get_deep_supervision_scales() # NOT USED

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # --- Transforms (Apply standard augmentation, NO DS downsampling on target) ---
        # Define transforms suitable for image-to-image tasks (adjust as needed)
        # Often simple resizing, random cropping (handled by dataloader), flipping
        def get_simple_transforms(patch_size, rotation_for_DA, mirror_axes, do_dummy_2d_data_aug):
            tr_transforms = []
            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert3DTo2DTransform())
                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size

            # Simpler spatial transform maybe? Or keep the complex one if needed.
            # Example: Just mirroring
            if mirror_axes is not None and len(mirror_axes) > 0:
                 tr_transforms.append(MirrorTransform(mirror_axes))

            # Maybe add basic color augmentation if applicable
            # tr_transforms.append(BrightnessMultiplicativeTransform(...))

            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert2DTo3DTransform())

            tr_transforms.append(RemoveLabelTransform(-1, 0)) # Keep if background label exists
            tr_transforms.append(RenameTransform('seg', 'target', True)) # Rename seg to target

            # CRITICAL: DO NOT Downsample target for CycleGAN
            # tr_transforms.append(DownsampleSegForDSTransform2(...)) # REMOVED

            # Convert to Tensor
            tr_transforms.append(NumpyToTensor(['data', 'target'], 'float')) # Only data and target needed usually
            return Compose(tr_transforms)

        # --- Create Transforms ---
        # Use the same transforms for both domains usually
        transforms_A = get_simple_transforms(patch_size, rotation_for_DA, mirror_axes, do_dummy_2d_data_aug)
        transforms_B = get_simple_transforms(patch_size, rotation_for_DA, mirror_axes, do_dummy_2d_data_aug)
        # Validation transforms might be simpler (e.g., only ToTensor)
        val_transforms_A = Compose([RemoveLabelTransform(-1, 0), RenameTransform('seg', 'target', True), NumpyToTensor(['data', 'target'], 'float')])
        val_transforms_B = Compose([RemoveLabelTransform(-1, 0), RenameTransform('seg', 'target', True), NumpyToTensor(['data', 'target'], 'float')])


        # --- Create Datasets (Requires adaptation based on your data structure) ---
        # This assumes you have ways to identify cases for Domain A and Domain B
        # Maybe different preprocessed folders? Or filter keys?
        # Example: Filter keys based on a naming convention or list
        all_keys = get_case_identifiers(self.preprocessed_dataset_folder)
        # Placeholder logic: split keys based on name prefix/suffix or predefined lists
        keys_A = [k for k in all_keys if k.startswith("DomainA_")] # EXAMPLE
        keys_B = [k for k in all_keys if k.startswith("DomainB_")] # EXAMPLE
        if not keys_A or not keys_B:
             raise ValueError("Could not find keys for both Domain A and Domain B based on example naming. Adapt get_dataloaders!")

        # Split keys for train/val (using the same fold logic as parent)
        tr_keys_all, val_keys_all = self.do_split() # Gets split based on parent's logic
        tr_keys_A = [k for k in keys_A if k in tr_keys_all]
        val_keys_A = [k for k in keys_A if k in val_keys_all]
        tr_keys_B = [k for k in keys_B if k in tr_keys_all]
        val_keys_B = [k for k in keys_B if k in val_keys_all]


        dataset_tr_A = nnDoseDataset(self.preprocessed_dataset_folder, tr_keys_A, num_images_properties_loading_threshold=0)
        dataset_val_A = nnDoseDataset(self.preprocessed_dataset_folder, val_keys_A, num_images_properties_loading_threshold=0)
        dataset_tr_B = nnDoseDataset(self.preprocessed_dataset_folder, tr_keys_B, num_images_properties_loading_threshold=0)
        dataset_val_B = nnDoseDataset(self.preprocessed_dataset_folder, val_keys_B, num_images_properties_loading_threshold=0)

        # --- Create Dataloaders (Using plain dataloaders from parent) ---
        # Note: CycleGAN doesn't use oversampling typically. Need to adjust DataLoader args?
        # Let's use the parent's loader class for now, assuming it can work without oversampling if needed.
        # Set oversample_foreground_percent=0 maybe?
        oversample_percent = 0.0 # Disable oversampling for CycleGAN

        common_dl_kwargs = {
            'batch_size': self.batch_size, # Use batch size calculated by parent
            'initial_patch_size': initial_patch_size,
            'patch_size': self.configuration_manager.patch_size,
            'label_manager': self.label_manager, # Use label manager for structure
            'oversample_foreground_percent': oversample_percent, # Disable oversampling
            'sampling_probabilities': None,
            'pad_sides': None
        }
        common_val_dl_kwargs = {
             'batch_size': self.batch_size,
             'patch_size': self.configuration_manager.patch_size, # Val uses final patch size directly
             'label_manager': self.label_manager,
             'oversample_foreground_percent': oversample_percent,
             'sampling_probabilities': None,
             'pad_sides': None
        }

        DataLoaderClass = CycleGANDataLoader3D # Use the new class

        dl_tr_A = DataLoaderClass(dataset_tr_A, **common_dl_kwargs, force_random_patch=True) # Add force_random_patch=True
        dl_val_A = DataLoaderClass(dataset_val_A, **common_val_dl_kwargs, force_random_patch=True) # Use for val too for consistency
        dl_tr_B = DataLoaderClass(dataset_tr_B, **common_dl_kwargs, force_random_patch=True)
        dl_val_B = DataLoaderClass(dataset_val_B, **common_val_dl_kwargs, force_random_patch=True)


        # --- Setup Multithreaded Augmenters ---
        allowed_num_processes = get_allowed_n_proc_DA()
        self.print_to_log_file("CycleGAN: Using", allowed_num_processes, "CPU processes for DA")

        if allowed_num_processes == 0:
            mt_gen_train_A = SingleThreadedAugmenter(dl_tr_A, transforms_A)
            mt_gen_val_A = SingleThreadedAugmenter(dl_val_A, val_transforms_A)
            mt_gen_train_B = SingleThreadedAugmenter(dl_tr_B, transforms_B)
            mt_gen_val_B = SingleThreadedAugmenter(dl_val_B, val_transforms_B)
        else:
            num_cached_tr = 6
            num_cached_val = 3
            # Train loaders need infinite length for itertools.cycle later
            mt_gen_train_A = torch.utils.data.DataLoader(dl_tr_A, num_workers=allowed_num_processes, pin_memory=self.device.type == 'cuda', batch_size=None, prefetch_factor=2) # Use PyTorch DataLoader maybe? Or adapt LimitedLenWrapper
            mt_gen_train_B = torch.utils.data.DataLoader(dl_tr_B, num_workers=allowed_num_processes, pin_memory=self.device.type == 'cuda', batch_size=None, prefetch_factor=2)
            # Keep val loaders finite
            mt_gen_val_A = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val_A, transform=val_transforms_A, num_processes=max(1, allowed_num_processes // 2), num_cached=num_cached_val, seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val_B = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val_B, transform=val_transforms_B, num_processes=max(1, allowed_num_processes // 2), num_cached=num_cached_val, seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.02)
            # NOTE: Using PyTorch DataLoader for training assumes your nnDoseDataLoader3D acts like a map-style dataset. If it's iterable, you need a different approach or adapt LimitedLenWrapper for infinite iteration. Let's store the base loaders for now.
            self.dataloader_tr_A_base = dl_tr_A
            self.dataloader_tr_B_base = dl_tr_B
            self.dataloader_val_A_aug = mt_gen_val_A
            self.dataloader_val_B_aug = mt_gen_val_B


        # Store the augmented dataloaders (or iterators for train)
        # We will create iterators in run_training
        # self.dataloader_train = None # Handled differently
        self.dataloader_val = None # Handled differently

        self.print_to_log_file("CycleGAN Dataloaders initialization done (Adaptation likely needed).")

        # Return None, None because the main run_training loop expects specific structure
        # We will manage dataloading within the overridden run_training
        return None, None
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnDoseNet. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_dsv:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        if hasattr(mod, 'decoder'):
            mod.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)#self.enable_deep_supervision

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_gt=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_dsv:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        #we need structure.json for evaluation
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'structure.json'),
                    join(self.output_folder_base, 'structure.json'))

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")
    def train_step(self, batch_A: dict, batch_B: dict) -> dict:
        """ Overrides parent. Implements the CycleGAN training logic for one iteration."""
        real_A = batch_A['data'].to(self.device, non_blocking=True)
        real_B = batch_B['data'].to(self.device, non_blocking=True)
        # Target is not directly used in CycleGAN loss calculation like in segmentation

        # --- Generate Fake Images ---
        # Need to access non-DDP model for generation if wrapped? No, DDP handles forward.
        fake_B = self.G_AtoB(real_A)
        fake_A = self.G_BtoA(real_B)

        # == Train Generators ==
        set_requires_grad([self.D_A, self.D_B], False) # Ds require no gradients now
        self.optimizer_G.zero_grad()

        # 1. Adversarial Loss (make Ds think fakes are real)
        loss_G_AtoB = self.criterion_GAN(self.D_B(fake_B), torch.tensor(1.0).expand_as(self.D_B(fake_B)).to(self.device))
        loss_G_BtoA = self.criterion_GAN(self.D_A(fake_A), torch.tensor(1.0).expand_as(self.D_A(fake_A)).to(self.device))
        loss_G_adv = loss_G_AtoB + loss_G_BtoA

        # 2. Cycle Consistency Loss
        reconstructed_A = self.G_BtoA(fake_B)
        loss_cycle_A = self.criterion_Cycle(reconstructed_A, real_A) * self.lambda_cycle
        reconstructed_B = self.G_AtoB(fake_A)
        loss_cycle_B = self.criterion_Cycle(reconstructed_B, real_B) * self.lambda_cycle
        loss_cycle = loss_cycle_A + loss_cycle_B

        # 3. Identity Loss (Optional)
        loss_identity = 0
        if self.lambda_identity > 0:
             # G_AtoB should be identity if real_B is passed
             identity_B = self.G_AtoB(real_B)
             loss_identity_B = self.criterion_Identity(identity_B, real_B) * self.lambda_identity
             # G_BtoA should be identity if real_A is passed
             identity_A = self.G_BtoA(real_A)
             loss_identity_A = self.criterion_Identity(identity_A, real_A) * self.lambda_identity
             loss_identity = loss_identity_A + loss_identity_B

        # Sum Generator losses
        loss_G = loss_G_adv + loss_cycle + loss_identity

        # Backward and Optimize G
        # Handle gradient scaling if enabled
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_G).backward()
            # Maybe clip gradients for G?
            # self.grad_scaler.unscale_(self.optimizer_G)
            # torch.nn.utils.clip_grad_norm_(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()), 1.0)
            self.grad_scaler.step(self.optimizer_G)
            # Scaler update happens once per iteration, maybe after D step? Let's put it here for now.
            # self.grad_scaler.update() # See below
        else:
            loss_G.backward()
            # Maybe clip gradients for G?
            # torch.nn.utils.clip_grad_norm_(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()), 1.0)
            self.optimizer_G.step()


        # == Train Discriminators ==
        set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()

        # --- Train D_A ---
        pred_real_A = self.D_A(real_A)
        loss_D_A_real = self.criterion_GAN(pred_real_A, torch.tensor(1.0).expand_as(pred_real_A).to(self.device))
        # Use detached fake_A
        pred_fake_A = self.D_A(fake_A.detach())
        loss_D_A_fake = self.criterion_GAN(pred_fake_A, torch.tensor(0.0).expand_as(pred_fake_A).to(self.device))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        # Backward D_A
        if self.grad_scaler is not None:
             self.grad_scaler.scale(loss_D_A).backward() # Scale D losses individually
        else:
             loss_D_A.backward()

        # --- Train D_B ---
        pred_real_B = self.D_B(real_B)
        loss_D_B_real = self.criterion_GAN(pred_real_B, torch.tensor(1.0).expand_as(pred_real_B).to(self.device))
        # Use detached fake_B
        pred_fake_B = self.D_B(fake_B.detach())
        loss_D_B_fake = self.criterion_GAN(pred_fake_B, torch.tensor(0.0).expand_as(pred_fake_B).to(self.device))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
         # Backward D_B
        if self.grad_scaler is not None:
             self.grad_scaler.scale(loss_D_B).backward()
        else:
             loss_D_B.backward()

        # Optimize D (updates both D_A and D_B based on their gradients)
        if self.grad_scaler is not None:
            # Unscale *before* clipping and stepping D
            self.grad_scaler.unscale_(self.optimizer_D)
            # Clip gradients for D?
            # torch.nn.utils.clip_grad_norm_(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), 1.0)
            self.grad_scaler.step(self.optimizer_D)
            # Update scaler ONCE per iteration, after both G and D steps
            self.grad_scaler.update()
        else:
            # Clip gradients for D?
            # torch.nn.utils.clip_grad_norm_(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), 1.0)
            self.optimizer_D.step()


        # --- Return loss dictionary for logging ---
        return {'loss_G': loss_G.detach().cpu().numpy(),
                'loss_G_adv': loss_G_adv.detach().cpu().numpy(),
                'loss_cycle': loss_cycle.detach().cpu().numpy(),
                'loss_identity': loss_identity.detach().cpu().numpy() if self.lambda_identity > 0 else 0.0,
                'loss_D_A': loss_D_A.detach().cpu().numpy(),
                'loss_D_B': loss_D_B.detach().cpu().numpy()
               }


    def validation_step(self, batch_A: dict, batch_B: dict) -> dict:
        """
        Overrides parent. Performs minimal validation for CycleGAN,
        typically calculating cycle loss or just preparing for qualitative eval.
        """
        real_A = batch_A['data'].to(self.device, non_blocking=True)
        real_B = batch_B['data'].to(self.device, non_blocking=True)

        # Generate images (no_grad context is handled by run_training loop)
        fake_B = self.G_AtoB(real_A)
        fake_A = self.G_BtoA(real_B)
        reconstructed_A = self.G_BtoA(fake_B)
        reconstructed_B = self.G_AtoB(fake_A)

        # Calculate Cycle Consistency Loss for validation
        loss_cycle_A = self.criterion_Cycle(reconstructed_A, real_A) # No lambda weighting needed for monitoring
        loss_cycle_B = self.criterion_Cycle(reconstructed_B, real_B)
        val_cycle_loss = loss_cycle_A + loss_cycle_B

        # Could also calculate identity loss if desired
        # Could also calculate adversarial loss using D, but less common for val metric

        return {'val_cycle_loss': val_cycle_loss.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """ Overrides parent. Aggregates and logs validation cycle loss. """
        outputs_collated = collate_outputs(val_outputs) # Collates the list of dicts

        # Aggregate cycle loss
        if 'val_cycle_loss' in outputs_collated:
            # Handle DDP aggregation if needed
            if self.is_dsv:
                world_size = dist.get_world_size()
                losses_val = [None for _ in range(world_size)]
                # Ensure the metric exists and is convertible to numpy/list for gather
                metric_to_gather = outputs_collated['val_cycle_loss']
                if isinstance(metric_to_gather, torch.Tensor):
                     metric_to_gather = metric_to_gather.cpu().numpy().tolist()
                elif isinstance(metric_to_gather, np.ndarray):
                     metric_to_gather = metric_to_gather.tolist()
                # Ensure it's a list of numbers if needed by all_gather_object
                if not isinstance(metric_to_gather, list):
                     metric_to_gather = [metric_to_gather]

                try:
                    dist.all_gather_object(losses_val, metric_to_gather)
                    # Careful aggregation: flatten and mean
                    flat_list = [item for sublist in losses_val for item in sublist]
                    loss_here = np.mean(flat_list)
                except Exception as e:
                     self.print_to_log_file(f"Error during DDP gather in validation: {e}")
                     loss_here = np.nan # Indicate failure

            else:
                loss_here = np.mean(outputs_collated['val_cycle_loss'])

            self.logger.log('val_cycle_loss', loss_here, self.current_epoch)
            self.print_to_log_file(f"Validation Cycle Loss: {np.round(loss_here, decimals=4)}")
        else:
             self.print_to_log_file("Validation output key 'val_cycle_loss' not found.")

        # Optional: Add qualitative evaluation (save sample images)
        if self.current_epoch % self.save_every == 0 or self.current_epoch == self.num_epochs -1 : # Every N epochs
            if self.local_rank == 0: # Only rank 0 saves images
                self.save_sample_images(self.current_epoch)


    def save_sample_images(self, epoch):
        """ Generates and saves a few sample image translations. """
        self.print_to_log_file(f"Saving sample images for epoch {epoch}...")
        # Get a fixed batch from validation loaders for consistent visualization
        try:
             # Need to handle the multi-threaded augmenters correctly
             # Maybe get samples directly from the base dataloader?
             val_loader_A = self.dataloader_val_A_aug._data_loader # Access base loader (adjust if structure differs)
             val_loader_B = self.dataloader_val_B_aug._data_loader # Access base loader
             fixed_batch_A = next(iter(val_loader_A))
             fixed_batch_B = next(iter(val_loader_B))

             real_A = fixed_batch_A['data'][0:min(4, self.batch_size)].to(self.device) # Take max 4 samples
             real_B = fixed_batch_B['data'][0:min(4, self.batch_size)].to(self.device)

             self.G_AtoB.eval()
             self.G_BtoA.eval()
             with torch.no_grad():
                fake_B = self.G_AtoB(real_A)
                fake_A = self.G_BtoA(real_B)
                reconstructed_A = self.G_BtoA(fake_B)
                reconstructed_B = self.G_AtoB(fake_A)
             self.G_AtoB.train() # Set back to train mode
             self.G_BtoA.train()

             # Save images (requires a function to convert tensor to image file, e.g., using nibabel/simpleitk)
             save_dir = join(self.output_folder, "validation_samples", f"epoch_{epoch}")
             maybe_mkdir_p(save_dir)

             # Example saving logic (needs a 'save_tensor_as_image' function)
             # for i in range(real_A.size(0)):
             #    save_tensor_as_image(real_A[i], join(save_dir, f"real_A_{i}.nii.gz"))
             #    save_tensor_as_image(fake_B[i], join(save_dir, f"fake_B_{i}.nii.gz"))
             #    save_tensor_as_image(reconstructed_A[i], join(save_dir, f"reconstructed_A_{i}.nii.gz"))
             #    save_tensor_as_image(real_B[i], join(save_dir, f"real_B_{i}.nii.gz"))
             #    save_tensor_as_image(fake_A[i], join(save_dir, f"fake_A_{i}.nii.gz"))
             #    save_tensor_as_image(reconstructed_B[i], join(save_dir, f"reconstructed_B_{i}.nii.gz"))

             self.print_to_log_file(f"Saved sample images to {save_dir}")

        except Exception as e:
            self.print_to_log_file(f"Could not save sample images: {e}")


    def on_train_epoch_end(self, train_outputs: List[dict]):
        """ Overrides parent. Aggregates and logs CycleGAN training losses. """
        outputs_collated = collate_outputs(train_outputs)

        # Aggregate and log each loss component
        loss_keys = ['loss_G', 'loss_G_adv', 'loss_cycle', 'loss_identity', 'loss_D_A', 'loss_D_B']
        for key in loss_keys:
            if key in outputs_collated:
                 # Handle DDP aggregation
                 if self.is_dsv:
                     try:
                         values = [None for _ in range(dist.get_world_size())]
                         # Ensure values are gatherable (list of numbers)
                         metric_to_gather = outputs_collated[key]
                         if isinstance(metric_to_gather, torch.Tensor):
                             metric_to_gather = metric_to_gather.cpu().numpy().tolist()
                         elif isinstance(metric_to_gather, np.ndarray):
                             metric_to_gather = metric_to_gather.tolist()
                         if not isinstance(metric_to_gather, list):
                              metric_to_gather = [metric_to_gather]

                         dist.all_gather_object(values, metric_to_gather)
                         flat_list = [item for sublist in values for item in sublist]
                         loss_here = np.mean(flat_list)
                     except Exception as e:
                         self.print_to_log_file(f"Error during DDP gather for {key}: {e}")
                         loss_here = np.nan
                 else:
                     loss_here = np.mean(outputs_collated[key])

                 self.logger.log(f'train_{key}', loss_here, self.current_epoch)
                 # Optional: print main losses
                 if key in ['loss_G', 'loss_D_A', 'loss_D_B']:
                      self.print_to_log_file(f'train_{key}', np.round(loss_here, decimals=4))

    def save_checkpoint(self, filename: str) -> None:
        """ Overrides parent. Saves state for all networks and optimizers. """
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                # Get state dicts, handling DDP and potential torch.compile wrapping
                def get_state_dict(network):
                     if self.is_dsv:
                         mod = network.module
                     else:
                         mod = network
                     if isinstance(mod, OptimizedModule):
                         mod = mod._orig_mod
                     return mod.state_dict()

                checkpoint = {
                    'G_AtoB_weights': get_state_dict(self.G_AtoB),
                    'G_BtoA_weights': get_state_dict(self.G_BtoA),
                    'D_A_weights': get_state_dict(self.D_A),
                    'D_B_weights': get_state_dict(self.D_B),
                    'optimizer_G_state': self.optimizer_G.state_dict(),
                    'optimizer_D_state': self.optimizer_D.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema, # Might track best cycle loss instead?
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs, # Save original init args
                    'trainer_name': self.__class__.__name__,
                    # 'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes, # Keep if needed for gen inference
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """ Overrides parent. Loads state for all networks and optimizers. """
        if not self.was_initialized:
            # Important: Initialize needs the correct domain A/B jsons from init_args
            # Ensure my_init_kwargs is loaded *before* calling initialize if loading from file
            if isinstance(filename_or_checkpoint, str):
                 temp_checkpoint = torch.load(filename_or_checkpoint, map_location='cpu') # Load to CPU first
                 self.my_init_kwargs = temp_checkpoint['init_args']
                 # Re-call __init__ with loaded args to set up paths etc. correctly
                 # This assumes __init__ can be called multiple times or reset internal state
                 # Or, manually set necessary attributes from my_init_kwargs here.
                 # For simplicity, let's assume initialize handles it based on self.my_init_kwargs
                 super().__init__(**self.my_init_kwargs) # Re-run parent init

            self.initialize() # Now initialize networks etc.

        if isinstance(filename_or_checkpoint, str):
            self.print_to_log_file(f"Loading checkpoint: {filename_or_checkpoint}")
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        else:
            checkpoint = filename_or_checkpoint

        # Helper to load state dict, handling DDP/compile wrappers
        def load_state_dict_helper(network, state_dict):
            # Remove 'module.' prefix if necessary
            new_state_dict = {}
            has_module_prefix = list(state_dict.keys())[0].startswith('module.')
            is_ddp_wrapped = isinstance(network, DDP)

            for k, v in state_dict.items():
                name = k
                if is_ddp_wrapped and not has_module_prefix:
                     name = 'module.' + k # Add 'module.' if loading non-DDP weights into DDP model
                elif not is_ddp_wrapped and has_module_prefix:
                     name = k[7:] # Remove 'module.' if loading DDP weights into non-DDP model
                new_state_dict[name] = v

            # Handle potential torch.compile wrapper
            if isinstance(network, OptimizedModule):
                 network._orig_mod.load_state_dict(new_state_dict)
            elif isinstance(network, DDP) and isinstance(network.module, OptimizedModule):
                 network.module._orig_mod.load_state_dict(new_state_dict)
            elif isinstance(network, DDP):
                 network.module.load_state_dict(new_state_dict)
            else:
                 network.load_state_dict(new_state_dict)


        # Load network weights
        load_state_dict_helper(self.G_AtoB, checkpoint['G_AtoB_weights'])
        load_state_dict_helper(self.G_BtoA, checkpoint['G_BtoA_weights'])
        load_state_dict_helper(self.D_A, checkpoint['D_A_weights'])
        load_state_dict_helper(self.D_B, checkpoint['D_B_weights'])

        # Load optimizer states
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])

        # Load other states
        if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint.get('_best_ema', None) # Use .get for safety
        self.current_epoch = checkpoint.get('current_epoch', 0) # Use .get

        self.print_to_log_file(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")


    def run_training(self):
        """ Overrides parent. Implements the main CycleGAN training loop. """
        self.on_train_start() # Sets up folders, unpacks, initializes networks

        # --- Need iterators for domain A and B dataloaders ---
        # This assumes self.dataloader_tr_A_base and self.dataloader_tr_B_base are set in get_dataloaders
        # And that they are iterable datasets suitable for PyTorch DataLoader
        # Adjust num_workers as needed
        num_workers = get_allowed_n_proc_DA()
        train_loader_A = torch.utils.data.DataLoader(self.dataloader_tr_A_base, batch_size=None, # Batching done by nnDoseDataLoader
                                                     num_workers=num_workers, pin_memory=self.device.type == 'cuda',
                                                     persistent_workers=num_workers > 0)
        train_loader_B = torch.utils.data.DataLoader(self.dataloader_tr_B_base, batch_size=None,
                                                     num_workers=num_workers, pin_memory=self.device.type == 'cuda',
                                                     persistent_workers=num_workers > 0)

        # Use itertools.cycle to loop indefinitely over the shorter dataset if lengths differ
        train_iterator_A = iter(itertools.cycle(train_loader_A))
        train_iterator_B = iter(itertools.cycle(train_loader_B))

        # Determine iterations per epoch based on the *longer* dataset if lengths differ significantly,
        # or stick to the predefined self.num_iterations_per_epoch
        # iterations_this_epoch = max(len(train_loader_A), len(train_loader_B)) # If using standard loaders
        iterations_this_epoch = self.num_iterations_per_epoch # Use predefined length

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start() # Sets network.train(), logs epoch start

            # --- Training Loop ---
            self.print_to_log_file(f'Epoch {self.current_epoch}')
            # No LR scheduler step per epoch here (unless using epoch-based scheduler)
            # Log LRs at start
            self.print_to_log_file(f"LR G: {np.round(self.optimizer_G.param_groups[0]['lr'], decimals=6)}, LR D: {np.round(self.optimizer_D.param_groups[0]['lr'], decimals=6)}")

            train_outputs = []
            for batch_id in range(iterations_this_epoch):
                try:
                    batch_A = next(train_iterator_A)
                    batch_B = next(train_iterator_B)
                except StopIteration:
                    # Should not happen with itertools.cycle, but safety check
                    self.print_to_log_file("Dataloader StopIteration encountered unexpectedly.")
                    break
                except Exception as e:
                     self.print_to_log_file(f"Error fetching batch {batch_id}: {e}")
                     # Maybe skip batch or break?
                     continue

                # Perform the combined train step
                step_output = self.train_step(batch_A, batch_B)
                train_outputs.append(step_output)

                # --- LR SCHEDULING (Example: Linear Decay) ---
                # Implement linear decay after N epochs if desired
                # epochs_decay_start = self.num_epochs // 2
                # if epoch >= epochs_decay_start:
                #    decay_factor = (self.num_epochs - epoch) / (self.num_epochs - epochs_decay_start)
                #    new_lr_g = self.lr_g * decay_factor
                #    new_lr_d = self.lr_d * decay_factor
                #    for param_group in self.optimizer_G.param_groups:
                #        param_group['lr'] = new_lr_g
                #    for param_group in self.optimizer_D.param_groups:
                #        param_group['lr'] = new_lr_d
                # --- End LR Scheduling ---


            # Log final LRs for the epoch
            self.logger.log('lr_G', self.optimizer_G.param_groups[0]['lr'], self.current_epoch)
            self.logger.log('lr_D', self.optimizer_D.param_groups[0]['lr'], self.current_epoch)

            self.on_train_epoch_end(train_outputs) # Aggregates and logs training losses

            # --- Validation Loop ---
            with torch.no_grad():
                self.on_validation_epoch_start() # Sets network.eval()

                # Need iterators for validation dataloaders
                # These should NOT cycle and have finite length
                # Assume self.dataloader_val_A_aug and self.dataloader_val_B_aug are set
                val_iterator_A = iter(self.dataloader_val_A_aug)
                val_iterator_B = iter(self.dataloader_val_B_aug)
                val_outputs = []
                num_val_batches = min(len(self.dataloader_val_A_aug), len(self.dataloader_val_B_aug)) # Use shorter length

                for batch_id in range(num_val_batches): # Iterate fixed number of times
                    try:
                        val_batch_A = next(val_iterator_A)
                        val_batch_B = next(val_iterator_B)
                    except StopIteration:
                        break # Stop if one loader finishes early
                    except Exception as e:
                         self.print_to_log_file(f"Error fetching validation batch {batch_id}: {e}")
                         continue

                    val_outputs.append(self.validation_step(val_batch_A, val_batch_B))

                self.on_validation_epoch_end(val_outputs) # Aggregates and logs validation loss

            self.on_epoch_end() # Saves checkpoint, logs summary, increments epoch

        self.on_train_end() # Final checkpoint save, cleanup


    # --- Methods likely NOT needed or needing complete overhaul for CycleGAN ---
    # def _build_loss(self): # Already overridden
    # def plot_network_architecture(self): # Needs adaptation for 4 networks
    # def perform_actual_validation(self): # Inference and metrics are different
    # def set_deep_supervision_enabled(self, enabled: bool): # Not applicable to CycleGAN training loop