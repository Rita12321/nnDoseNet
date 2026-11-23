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

from nndosenet.configuration import ANISO_THRESHOLD, default_num_processes
from nndosenet.evaluation.evaluate_predictions import compute_dose_metrics_on_folder
from nndosenet.inference.export_prediction import export_prediction_from_logits
from nndosenet.inference.predict_from_raw_data import nnDoseNetPredictor,nnDiffusePredictor
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
from nndosenet.training.dataloading.data_loader_2d import nnDoseDataLoader2D
from nndosenet.training.dataloading.data_loader_3d import nnDoseDataLoader3D
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

class DiffuseTrainer(nnDoseTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        self.is_dsv = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_dsv else dist.get_rank()
        self.device = device

        if self.is_dsv:
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}. Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")


        self.my_init_kwargs = {k: locals()[k] for k in inspect.signature(self.__init__).parameters.keys()}

        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json

        self.fold = fold
        self.unpack_dataset = unpack_dataset
        self.num_epochs = self.configuration_manager.epoch_num
        self.mask_channel = self.configuration_manager.mask_channel
        self.PTV_OAR_BODY = self.configuration_manager.PTV_OAR_BODY

        self.preprocessed_dataset_folder_base = join(nnDose_preprocessed, self.plans_manager.dataset_name) \
            if nnDose_preprocessed is not None else None
        self.raw_dataset_folder_base = join(nnDose_raw, self.plans_manager.dataset_name) \
            if nnDose_raw is not None else None
        self.output_folder_base = join(nnDose_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnDose_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        self.num_timesteps =1000
        self.time_embedded_dim =128
        self.initial_lr = 1e-4#1e-2
        self.weight_decay = 3e-4 #3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.current_epoch = 0

        self.label_manager = self.plans_manager.get_label_manager(dataset_json)

        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        self.dataloader_train = self.dataloader_val = None
        self._best_ema = None
        self.inference_allowed_mirroring_axes = None
        self.save_every = 50
        self.disable_checkpointing = False

        self._set_batch_size_and_oversample()
        self.was_initialized = False

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)
            if ('nnDose_compile' in os.environ.keys()) and (
                    os.environ['nnDose_compile'].lower() in ('true', '1', 't')):
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_dsv:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _save_debug_information(self):
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                       num_input_channels, deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_dsv:
            self.batch_size = self.configuration_manager.batch_size
        else:
            batch_sizes = []
            oversample_percents = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

                sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
                sample_id_high = np.sum(batch_sizes)

                if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                    oversample_percents.append(0.0)
                elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                    oversample_percents.append(1.0)
                else:
                    percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
                    oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                    sample_id_low / global_batch_size) / percent_covered_by_this_rank)
                    oversample_percents.append(oversample_percent_here)

            self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]
            self.oversample_foreground_percent = oversample_percents[my_rank]

    def _build_loss(self):
        if (self.configuration_manager.dose == True) or self.label_manager.dose_label:
            channel_ptv_oar = [self.mask_channel.index(int(self.PTV_OAR_BODY[0])), self.mask_channel.index(int(self.PTV_OAR_BODY[1]))]

            if int(self.PTV_OAR_BODY[2]) < 0:
                dose_mask = None
            else:
                dose_mask = self.mask_channel.index(int(self.PTV_OAR_BODY[2]))

            self.print_to_log_file("Order of Target, OAR, Body in mask_channel:", channel_ptv_oar[0], channel_ptv_oar[1], dose_mask)

            self.loss_mode = str(self.configuration_manager.loss_mode)
            if self.configuration_manager.mask_channel is not None:
                if self.loss_mode == '0':
                    loss = Err_and_DVH_Loss(losses=[MSELoss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 0: 1*MSE")
                elif self.loss_mode == '1':
                    loss = Err_and_DVH_Loss(losses=[MSELoss, value_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv}], weights=[0.5, 0.5], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 1: 0.5*MSE+0.5*vDVH")
                elif self.loss_mode == '2':
                    loss = Err_and_DVH_Loss(losses=[MSELoss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv}], weights=[0.5, 0.5], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 2: 0.5*MSE+0.5*cDVH")
                elif self.loss_mode == '3':
                    loss = Err_and_DVH_Loss(losses=[MSELoss, value_DVH_Loss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv},{'dsv': self.is_dsv}], weights=[0.4, 0.3, 0.3], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 3: 0.4*MSE+0.3*vDVH+0.3*cDVH")
                elif self.loss_mode == '4':
                    loss = Err_and_DVH_Loss(losses=[MSELoss, value_DVH_Loss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv},{'dsv': self.is_dsv}], weights=[0.6, 0.2, 0.2], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 4: 0.6*MSE+0.2*vDVH+0.2*cDVH")
                elif self.loss_mode == '5':
                    loss = Err_and_DVH_Loss(losses=[L1Loss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 5: 1*MAE")
                elif self.loss_mode == '6':
                    loss = Err_and_DVH_Loss(losses=[L1Loss, value_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv}], weights=[0.5, 0.5], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 6: 0.5*MAE+0.5*vDVH")
                elif self.loss_mode == '7':
                    loss = Err_and_DVH_Loss(losses=[L1Loss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv}], weights=[0.5, 0.5], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 7: 0.5*MSE+0.5*cDVH")
                elif self.loss_mode == '8':
                    loss = Err_and_DVH_Loss(losses=[L1Loss, value_DVH_Loss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv},{'dsv': self.is_dsv}], weights=[0.4, 0.3, 0.3], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 8: 0.4*MAE+0.3*vDVH+0.3*cDVH")
                elif self.loss_mode == '9':
                    loss = Err_and_DVH_Loss(losses=[L1Loss, value_DVH_Loss, criteria_DVH_Loss], kwargs=[{},{'dsv': self.is_dsv},{'dsv': self.is_dsv}], weights=[0.6, 0.2, 0.2], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                    self.print_to_log_file(loss)
                    self.print_to_log_file("Using loss mode 9: 0.6*MAE+0.2*vDVH+0.2*cDVH")
                else:
                    self.print_to_log_file("unknown loss mode, set to default: MSE")
                    loss = Err_and_DVH_Loss(losses=[MSELoss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
            else:
                if self.loss_mode == '0':
                    loss = Err_and_DVH_Loss(losses=[MSELoss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                elif self.loss_mode == '5':
                    loss = Err_and_DVH_Loss(losses=[L1Loss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)
                else:
                    self.print_to_log_file("there aren't any masked channel(for PTV and OAR), so you can't use DVH based loss. The only option left is mode 0:MSE(default) and 5:MAE(L1Loss)")
                    self.print_to_log_file("set to default loss: MSE")
                    loss = Err_and_DVH_Loss(losses=[MSELoss], kwargs=[{}], weights=[1], channel_ptv_oar=channel_ptv_oar, dose_mask=dose_mask)

        else:
            self.print_to_log_file("Somehow it's not running in dose mode. Please check did you")
            self.print_to_log_file("1. put dose as label in dataset.json")
            self.print_to_log_file("2. Or put --dose argument when planning")

        deep_supervision_scales = self._get_deep_supervision_scales()

        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = ("%s:" % dt_object, *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)


    def configure_optimizers(self):


        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)

        optimizer = torch.optim.AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        # # --- Configuration for Linear Warmup + Cosine Decay ---
        # # Choose the number of warmup steps (iterations)
        # # Example: Warm up for 500 iterations or first 2 epochs
        # num_warmup_iterations = 500
        # # num_warmup_iterations = 2 * self.num_iterations_per_epoch # Alternative based on epochs

        # # Calculate total training steps (iterations)
        # num_training_iterations = self.num_epochs * self.num_iterations_per_epoch

        # self.print_to_log_file(f"Using AdamW optimizer with LR={self.initial_lr}, WD={self.weight_decay}")
        # self.print_to_log_file(f"LR Scheduler: Linear warmup for {num_warmup_iterations} iterations, then Cosine Decay over {num_training_iterations} total iterations.")
        # # ------------------------------------------------------

        # def lr_lambda(current_step: int):
        #     """
        #     Calculates the learning rate multiplier based on linear warmup
        #     followed by cosine decay.
        #     """
        #     # 1. Linear Warmup phase
        #     if current_step < num_warmup_iterations:
        #         return float(current_step) / float(max(1, num_warmup_iterations))

        #     # 2. Cosine Decay phase
        #     progress = float(current_step - num_warmup_iterations) / float(max(1, num_training_iterations - num_warmup_iterations))
        #     # Cosine decay formula from 1 to 0
        #     cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        #     # You could add a minimum LR factor here if needed, e.g.,
        #     # min_lr_factor = 0.01 # Decay to 1% of initial_lr
        #     # return (1.0 - min_lr_factor) * cosine_decay + min_lr_factor
        #     return cosine_decay # Decay towards 0

        # # Create the LambdaLR scheduler
        # lr_scheduler = LambdaLR(optimizer, lr_lambda)

        return optimizer, lr_scheduler
    def plot_network_architecture(self):
        if self.local_rank == 0:
            try:
                from torchviz import make_dot
                data_rand = torch.rand((1, self.num_input_channels, *self.configuration_manager.patch_size),
                                                       device=self.device)
                batch_size = data_rand.shape[0]
                t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
                make_dot(tuple(self.network(data_rand,t))).render(
                    join(self.output_folder, "network_architecture.pdf"), format='pdf')
                self.optimizer.zero_grad()
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)
            finally:
                empty_cache(self.device)

    def do_split(self):
        if self.fold == "all":
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnDoseDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0)
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = nnDoseDataset(self.preprocessed_dataset_folder, tr_keys,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnDoseDataset(self.preprocessed_dataset_folder, val_keys,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=0, order_resampling_seg=0,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            mask_channel=self.configuration_manager.mask_channel, PTV_OAR_BODY=self.configuration_manager.PTV_OAR_BODY)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        mask_channel=self.configuration_manager.mask_channel)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        self.print_to_log_file("Using", allowed_num_processes, "CPU")
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)

        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnDoseDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnDoseDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnDoseDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnDoseDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 0,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                mask_channel=None,
                                PTV_OAR_BODY=None) -> AbstractTransform:

        tr_transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())
        if mask_channel is not None:
            if len(mask_channel) > 0:
                tr_transforms.append(SeperateDataMask(mask_channel=mask_channel, input_key='data', output_key='data_mask'))
        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='data_mask',
                                                              output_key='data_mask'))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=0, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform('seg', 'target', True))

        # if deep_supervision_scales is not None:
        #     tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
        #                                                       output_key='target'))
        # Inside get_training_transforms and get_validation_transforms
        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(
                deep_supervision_scales,
                order=1,  # <-- Use trilinear interpolation instead of nearest neighbor
                input_key='target',
                output_key='target'
            ))
        if mask_channel is not None:
            if len(mask_channel) > 0:
                tr_transforms.append(revert_SeperateDataMask(mask_channel=mask_channel, input_key='data', output_key='data_mask', dsv=(deep_supervision_scales is not None)))
        tr_transforms.append(NumpyToTensor(['data', 'target', 'data_mask'], 'float'))

        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                                  mask_channel=None) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        if mask_channel is not None:
            if len(mask_channel) > 0:
                val_transforms.append(CopyDataMask(mask_channel=mask_channel, input_key='data', output_key='data_mask'))
        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='data_mask',
                                                              output_key='data_mask'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target', 'data_mask'], 'float'))

        val_transforms = Compose(val_transforms)
        return val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_dsv:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_gt=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_dsv:
            dist.barrier()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        shutil.copy(join(self.preprocessed_dataset_folder_base, 'structure.json'),
                    join(self.output_folder_base, 'structure.json'))

        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        self.plot_network_architecture()

        self._save_debug_information()

    def on_train_end(self):
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        dsv_data = batch['data_mask']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if isinstance(dsv_data, list):
            dsv_data = [i.to(self.device, non_blocking=True) for i in dsv_data]
        else:
            dsv_data = dsv_data.to(self.device, non_blocking=True)

        # Example in DiffuseTrainer.train_step
        batch_size = data.shape[0]
        # Sample random timesteps, one per batch item
        t = torch.randint(0,self.num_timesteps, (batch_size,), device=data.device).long()
        # Now 't' has shape [batch_size]

        # Pass data and correctly shaped t to the network

        self.optimizer.zero_grad()

        output = self.network(data, t)
        # for i in range(len(target)):
        #     print(f"shape of target[{i}]:", target[i].shape)
        # for i in range(len(output)):
        #     print(f"shape of output[{i}]:", output[i].shape)
        l = self.loss(output, target, dsv_data)#

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)#12
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)#12
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        if self.is_dsv:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        self.logger.log('val_losses', loss_here, self.current_epoch)
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        mask_data = batch['data_mask']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if isinstance(mask_data, list):

            mask_data = [i.to(self.device, non_blocking=True) for i in mask_data]
        else:

            mask_data = mask_data.to(self.device, non_blocking=True)


        batch_size = data.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=data.device).long()
        output = self.network(data,t)

        l = self.loss(output, target,mask_data)#)
        del data



        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]


        return {'loss': l.detach().cpu().numpy()}
    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['val_losses'][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['val_losses'][-1]
            self.print_to_log_file(f"Yayy! New best validation_loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_dsv:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        if self.is_dsv:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnDiffusePredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_gpu=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False,num_timesteps=self.num_timesteps )
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_dsv:
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = nnDoseDataset(self.preprocessed_dataset_folder, val_keys,
                                        num_images_properties_loading_threshold=0)

            results = []
            for k in dataset_val.keys():
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                 allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                     allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, properties = dataset_val.load_case(k)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                if prediction.device != data.device:
                    data = data.to(prediction.device)
                if int(self.PTV_OAR_BODY[2]) > 0:
                    prediction = torch.clamp(prediction * data[int(self.PTV_OAR_BODY[2]), ...], min=0)

                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated),
                        )
                    )
                )
            _ = [r.get() for r in results]

        if self.is_dsv:
            dist.barrier()

        if self.local_rank == 0:
            try:
                structure_json = load_json(join(self.raw_dataset_folder_base, 'structure.json'))['structure_label']
            except:
                structure_json = None
            metrics = compute_dose_metrics_on_folder(folder_ref=join(self.preprocessed_dataset_folder_base, 'ground_truths'),
                                                folder_pred=validation_output_folder,
                                                folder_input=join(self.raw_dataset_folder_base, 'imagesTr'),
                                                output_file=join(validation_output_folder, 'summary.json'),
                                                image_reader_writer=self.plans_manager.image_reader_writer_class(),
                                                file_ending=self.dataset_json["file_ending"],
                                                structure_df=structure_json,
                                                dose_label=self.label_manager.dose_label,
                                                chill=True)
            self.print_to_log_file("Validation complete", also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
