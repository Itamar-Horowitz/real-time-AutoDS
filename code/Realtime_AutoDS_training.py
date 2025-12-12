import os
import math
import time
import json
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, zoom
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Configuration Class
# ============================================================================

class Config:
    """Configuration for AutoDS Training"""

    # Paths
    model_save_path = '../models/v3-depth-wise-convolution/v32'
    model_name = 'diff_1'
    tf_model_metadata_path = '../models/diff_1/model_metadata.mat'
    tf_training_report_path = '../models/diff_1/training_report.pdf'

    pretrained_weights_path = ""

    # Pruning Configuration
    use_pruning = False
    prune_ratio = 0.0
    sparsity_lambda = 0

    # Depthwise Separable Convolution Configuration
    use_depth_wise_conv = True  # Set to True to train with depthwise separable convolutions

    # Training Parameters
    number_of_epochs = 40
    batch_size = 16
    percentage_validation = 20
    initial_learning_rate = 0.001

    # Model Parameters (can be overwritten from metadata files)
    upsampling_factor = 8
    pixel_size = 107
    wavelength = 715
    numerical_aperture = 1.49
    L2_weighting_factor = 100

    # Simulation Parameters (can be read from training report PDF)
    FOV_size = 4280
    ADC_per_photon_conversion = 1
    ReadOutNoise_ADC = 20
    ADC_offset = 50
    emitter_density = 1.5
    emitter_density_std = 0.5
    number_of_frames = 100
    sigma = 124
    sigma_std = 20
    n_photons = 900
    n_photons_std = 100

    # Patch Generation Parameters
    patch_size = 30
    num_patches_per_frame = 10
    min_number_of_emitters_per_patch = 0
    max_num_patches = 10000
    gaussian_sigma = 1


config = Config()


# ============================================================================
# Metadata Reading Functions
# ============================================================================

def read_model_metadata_mat(mat_path):
    """Read model metadata from .mat file and update config."""
    print(f"\n{'=' * 70}")
    print(f"Reading model metadata from: {mat_path}")
    print(f"{'=' * 70}")

    mat_data = sio.loadmat(mat_path)
    read_params = {}

    key_mapping = {
        'upsampling_factor': 'upsampling_factor',
        'Normalization factor': 'L2_weighting_factor',
        'pixel_size': 'pixel_size',
        'wavelength': 'wavelength',
        'numerical_aperture': 'numerical_aperture',
    }

    for mat_key, config_attr in key_mapping.items():
        if mat_key in mat_data:
            value = mat_data[mat_key]
            if isinstance(value, np.ndarray):
                value = value.squeeze()
                if value.ndim == 0:
                    value = float(value)
            read_params[config_attr] = value
            setattr(config, config_attr, value)
            print(f"  {mat_key}: {value}")

    print(f"{'=' * 70}\n")
    return read_params


def read_training_report_pdf(pdf_path):
    """Read training parameters from PDF report and update config."""
    print(f"\n{'=' * 70}")
    print(f"Reading training report from: {pdf_path}")
    print(f"{'=' * 70}")

    text = ""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except ImportError:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    if not text:
        print("Warning: Could not extract text from PDF")
        return {}

    read_params = {}
    key_mapping = {
        'FOV size (nm)': ('FOV_size', float),
        'Pixel size (nm)': ('pixel_size', float),
        'ADC/photon': ('ADC_per_photon_conversion', float),
        'Read-out noise (ADC)': ('ReadOutNoise_ADC', float),
        'Constant offset (ADC)': ('ADC_offset', float),
        'Emitter density (emitters/um^2)': ('emitter_density', float),
        'STD of emitter density (emitters/um^2)': ('emitter_density_std', float),
        'Number of frames': ('number_of_frames', int),
        'Wavelength (nm)': ('wavelength', float),
        'NA': ('numerical_aperture', float),
        'Sigma (nm)': ('sigma', float),
        'STD of Sigma (nm)': ('sigma_std', float),
        'Number of photons': ('n_photons', float),
        'STD of number of photons': ('n_photons_std', float),
        'SNR': ('SNR', float),
        'STD of SNR': ('SNR_std', float),
        'patch size': ('patch_size', int),
        'upsampling factor': ('upsampling_factor', int),
        'num_patches_per_frame': ('num_patches_per_frame', int),
        'min_number_of_emitters_per_patch': ('min_number_of_emitters_per_patch', int),
        'max_num_patches': ('max_num_patches', int),
        'gaussian_sigma': ('gaussian_sigma', float),
        'L2 weighting factor': ('L2_weighting_factor', float),
    }

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                if key in key_mapping:
                    config_attr, dtype = key_mapping[key]
                    try:
                        value = dtype(value_str)
                        read_params[config_attr] = value
                        setattr(config, config_attr, value)
                        print(f"  {key}: {value}")
                    except ValueError:
                        print(f"  Warning: Could not parse value for {key}: {value_str}")

    print(f"{'=' * 70}\n")
    return read_params


def load_metadata_from_files():
    """Load metadata from both .mat and PDF files if paths are configured."""
    mat_params = {}
    pdf_params = {}
    if config.tf_model_metadata_path:
        mat_params = read_model_metadata_mat(config.tf_model_metadata_path)
    if config.tf_training_report_path:
        pdf_params = read_training_report_pdf(config.tf_training_report_path)
    return {**mat_params, **pdf_params}


def print_current_config():
    """Print current configuration values"""
    print(f"\n{'=' * 70}")
    print("CURRENT CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"\n[Paths]")
    print(f"  Model save path: {config.model_save_path}")
    print(f"  Model name: {config.model_name}")
    print(f"  Metadata .mat path: {config.tf_model_metadata_path}")
    print(f"  Training report PDF path: {config.tf_training_report_path}")
    print(f"\n[Pruning]")
    print(f"  use_pruning: {config.use_pruning}")
    print(f"  prune_ratio: {config.prune_ratio}")
    print(f"  sparsity_lambda: {config.sparsity_lambda}")
    print(f"\n[Depthwise Separable Convolution]")
    print(f"  use_depth_wise_conv: {config.use_depth_wise_conv}")
    print(f"\n[Training]")
    print(f"  number_of_epochs: {config.number_of_epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  percentage_validation: {config.percentage_validation}")
    print(f"  initial_learning_rate: {config.initial_learning_rate}")
    print(f"\n[Model Parameters]")
    print(f"  upsampling_factor: {config.upsampling_factor}")
    print(f"  pixel_size: {config.pixel_size} nm")
    print(f"  wavelength: {config.wavelength} nm")
    print(f"  numerical_aperture: {config.numerical_aperture}")
    print(f"  L2_weighting_factor: {config.L2_weighting_factor}")
    print(f"\n[Simulation Parameters]")
    print(f"  FOV_size: {config.FOV_size} nm")
    print(f"  emitter_density: {config.emitter_density} emitters/um^2")
    print(f"  number_of_frames: {config.number_of_frames}")
    print(f"  sigma: {config.sigma} nm")
    print(f"  n_photons: {config.n_photons}")
    print(f"  ReadOutNoise_ADC: {config.ReadOutNoise_ADC}")
    print(f"\n[Patch Generation]")
    print(f"  patch_size: {config.patch_size}")
    print(f"  num_patches_per_frame: {config.num_patches_per_frame}")
    print(f"  max_num_patches: {config.max_num_patches}")
    print(f"  gaussian_sigma: {config.gaussian_sigma}")
    print(f"{'=' * 70}\n")


# ============================================================================
# Check for GPU
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# Helper Functions
# ============================================================================

def gaussian_interpolation_batch(data_batch, scale, sigma=1):
    """
    Applies Gaussian interpolation (smoothing and upsampling) to a batch of images.
    """
    upsampled_data_batch = []
    for data in data_batch:
        smoothed_data = gaussian_filter(data, sigma=sigma)
        upsampled_data = zoom(smoothed_data, scale, order=3)
        upsampled_data_batch.append(upsampled_data)
    return np.array(upsampled_data_batch)


def project_01(im):
    im = np.squeeze(im)
    return (im - im.min()) / (im.max() - im.min())


def normalize_im(im, dmean, dstd):
    return (np.squeeze(im) - dmean) / dstd


def matlab_style_gauss2D(shape=(7, 7), sigma=1):
    """Create 2D Gaussian kernel matching MATLAB style"""
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel.astype(np.float32) / np.sum(kernel)


# Create Gaussian filter as a tensor
psf_heatmap = matlab_style_gauss2D(shape=(7, 7), sigma=1)
gfilter = torch.from_numpy(psf_heatmap).view(1, 1, 7, 7).float()


# ============================================================================
# CNN Building Blocks
# ============================================================================

class ConvBNReLU(nn.Module):
    """Conv2D + BatchNorm + ReLU block"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ConvBNReLU, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize with Orthogonal (similar to Keras)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================================
# Depthwise Separable Convolution Block (for faster inference)
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv

    This significantly reduces computation compared to standard convolution:
    - Standard Conv: in_ch * out_ch * k * k * H * W operations
    - Depthwise Separable: (in_ch * k * k + in_ch * out_ch) * H * W operations

    For k=3, in_ch=64, out_ch=128:
    - Standard: 64 * 128 * 9 = 73,728 params
    - Depthwise Separable: 64 * 9 + 64 * 128 = 8,768 params (~8.4x reduction)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise convolution (default: 3)
        stride: Stride for depthwise convolution (default: 1)
        padding: Padding (auto-calculated if None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(DepthwiseSeparableConv, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        # Depthwise convolution (spatial filtering per channel)
        # groups=in_channels means each input channel is convolved independently
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise convolution (1x1 conv for channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


# ============================================================================
# CNN Model with Upsampling - Supports standard, pruned, and depthwise architectures
# ============================================================================

class CNNUpsample(nn.Module):
    """
    CNN with upsampling for super-resolution.

    Supports multiple modes:
    1. Full architecture (default): Standard channel counts [32, 64, 128, 256, 128, 64]
    2. Pruned architecture: Custom channel counts specified via pruned_channels dict
    3. Depthwise separable: Uses depthwise separable convolutions for encoder/decoder layers

    Args:
        in_channels: Number of input channels (default: 1)
        upsampling_factor: Upsampling factor (default: 8)
        pruned_channels: Optional dict specifying pruned channel counts for each layer.
                         Format: {'F1': n1, 'F2': n2, ..., 'F6': n6}
                         If None, uses full (unpruned) architecture.
        use_depth_wise_conv: If True, uses depthwise separable convolutions for encoder/decoder
                       layers (conv_bn_relu1-6) for faster inference. Upsample blocks
                       always use standard convolutions. (default: False)
    """

    def __init__(self, in_channels=1, upsampling_factor=8, pruned_channels=None, use_depth_wise_conv=False):
        super(CNNUpsample, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.pruned_channels = pruned_channels
        self.use_depth_wise_conv = use_depth_wise_conv

        # Determine channel counts
        # NOTE: F6 (ch6) is always kept at 64 channels to preserve upsample_blocks unchanged
        if pruned_channels is not None:
            ch1 = pruned_channels.get('F1', 32)
            ch2 = pruned_channels.get('F2', 64)
            ch3 = pruned_channels.get('F3', 128)
            ch4 = pruned_channels.get('F4', 256)
            ch5 = pruned_channels.get('F5', 128)
            # F6 is NOT pruned - always use original 64 channels to keep upsample_blocks unchanged
            ch6 = 64  # Fixed - upsample_blocks expect 64 input channels
            self.is_pruned = True
        else:
            ch1, ch2, ch3, ch4, ch5, ch6 = 32, 64, 128, 256, 128, 64
            self.is_pruned = False

        # Select convolution block type for encoder/decoder layers
        ConvBlock = DepthwiseSeparableConv if use_depth_wise_conv else ConvBNReLU

        # Encoder with fused blocks (uses depthwise if enabled)
        self.conv_bn_relu1 = ConvBlock(in_channels, ch1, 3, 1)
        self.conv_bn_relu2 = ConvBlock(ch1, ch2, 3, 1)
        self.conv_bn_relu3 = ConvBlock(ch2, ch3, 3, 1)
        self.conv_bn_relu4 = ConvBlock(ch3, ch4, 3, 1)

        # Decoder with fused blocks (uses depthwise if enabled)
        self.conv_bn_relu5 = ConvBlock(ch4, ch5, 3, 1)
        # F6 output is always 64 channels to match upsample_blocks input
        self.conv_bn_relu6 = ConvBlock(ch5, ch6, 3, 1)

        # Upsampling blocks - ALWAYS use standard ConvBNReLU (not depthwise)
        # These blocks are NEVER pruned - always use original architecture
        num_upsample_blocks = int(np.log2(upsampling_factor))
        self.upsample_blocks = nn.ModuleList()

        for i in range(num_upsample_blocks):
            # First block takes 64 channels from F6, rest take 32 channels
            in_ch = 64 if i == 0 else 32  # Fixed input channels - upsample_blocks are unchanged
            block = nn.ModuleDict({
                'upsample': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                'conv_bn_relu': ConvBNReLU(in_ch, 32, 5, 1)  # Always standard conv, always 32 output
            })
            self.upsample_blocks.append(block)

        # Prediction head: use_bias=False and Orthogonal init to match TensorFlow
        self.prediction = nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=False)

        # Initialize only the prediction layer with Orthogonal (matching TensorFlow)
        self._initialize_prediction_weights()

    def _initialize_prediction_weights(self):
        """Initialize only the prediction layer with Orthogonal to match TensorFlow."""
        nn.init.orthogonal_(self.prediction.weight)

    def forward(self, x):
        # Encoder
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv_bn_relu4(x)

        # Decoder
        x = self.conv_bn_relu5(x)
        x = self.conv_bn_relu6(x)

        # Upsampling
        for block in self.upsample_blocks:
            x = block['upsample'](x)
            x = block['conv_bn_relu'](x)

        # Prediction
        x = self.prediction(x)
        return x

    def get_architecture_info(self):
        """Return architecture information for this model"""
        return {
            'upsampling_factor': self.upsampling_factor,
            'pruned_channels': self.pruned_channels,
            'is_pruned': self.is_pruned,
            'use_depth_wise_conv': self.use_depth_wise_conv
        }

    def get_channel_counts(self):
        """Return the channel counts for each layer"""
        if self.use_depth_wise_conv:
            return {
                'F1': self.conv_bn_relu1.pointwise.out_channels,
                'F2': self.conv_bn_relu2.pointwise.out_channels,
                'F3': self.conv_bn_relu3.pointwise.out_channels,
                'F4': self.conv_bn_relu4.pointwise.out_channels,
                'F5': self.conv_bn_relu5.pointwise.out_channels,
                'F6': self.conv_bn_relu6.pointwise.out_channels,
            }
        else:
            return {
                'F1': self.conv_bn_relu1.conv.out_channels,
                'F2': self.conv_bn_relu2.conv.out_channels,
                'F3': self.conv_bn_relu3.conv.out_channels,
                'F4': self.conv_bn_relu4.conv.out_channels,
                'F5': self.conv_bn_relu5.conv.out_channels,
                'F6': self.conv_bn_relu6.conv.out_channels,
            }


# ============================================================================
# Loss Function
# ============================================================================

class CustomLoss(nn.Module):
    """Custom loss for upsampling model with optional sparsity regularization"""

    def __init__(self, sparsity_lambda=0.0):
        super(CustomLoss, self).__init__()
        self.sparsity_lambda = sparsity_lambda
        self.register_buffer('gfilter', gfilter)

    def forward(self, y_pred, y_true, model=None):
        # Apply Gaussian convolution
        gf = self.gfilter.to(y_pred.device)
        # heatmap_pred = F.conv2d(y_pred, gf, padding=3)
        heatmap_pred = F.conv2d(y_pred, gf, stride=1, padding='same')

        # MSE on heatmaps
        loss_heatmaps = torch.mean((y_true - heatmap_pred) ** 2)

        # L1 on predictions (sparsity in output)
        loss_spikes = torch.mean(torch.abs(y_pred))

        total_loss = loss_heatmaps + loss_spikes

        # Add L1 regularization on conv weights if sparsity_lambda > 0
        if self.sparsity_lambda > 0 and model is not None:
            l1_reg = 0.0
            for name, param in model.named_parameters():
                # Handle both standard conv and depthwise separable conv
                if ('conv.weight' in name or 'depthwise.weight' in name or 'pointwise.weight' in name) \
                        and 'upsample' not in name:
                    l1_reg += torch.sum(torch.abs(param))
            total_loss = total_loss + self.sparsity_lambda * l1_reg

        return total_loss


# ============================================================================
# Training Callbacks
# ============================================================================

class ModelCheckpoint:
    """Save model when validation loss improves"""

    def __init__(self, filepath, save_best_only=True, verbose=True):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss = float('inf')

    def __call__(self, model, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: val_loss improved to {val_loss:.6f}, saving model")
            return True
        return False


class LossHistory:
    """Track training history"""

    def __init__(self):
        self.history = {'loss': [], 'val_loss': []}

    def append(self, loss, val_loss):
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)


# ============================================================================
# Pruning Functions
# ============================================================================

def compute_filter_importance(model, use_depth_wise_conv=False):
    """
    Compute importance score for each filter based on L1 norm of weights.
    Returns a dictionary mapping layer names to importance scores.

    For depthwise separable convolutions, importance is computed based on
    the pointwise (1x1) convolution weights, as they define the output channels.

    NOTE: conv_bn_relu6 (F6) is excluded from pruning to keep upsample_blocks unchanged.
    """
    importance_scores = {}

    # Only prune layers F1-F5, NOT F6 (to keep upsample_blocks unchanged)
    layer_names = ['conv_bn_relu1', 'conv_bn_relu2', 'conv_bn_relu3',
                   'conv_bn_relu4', 'conv_bn_relu5']  # F6 excluded

    for layer_name in layer_names:
        if hasattr(model, layer_name):
            block = getattr(model, layer_name)
            if use_depth_wise_conv:
                # For depthwise separable, use pointwise weights
                weights = block.pointwise.weight.data  # Shape: (out_ch, in_ch, 1, 1)
            else:
                # For standard conv, use conv weights
                weights = block.conv.weight.data  # Shape: (out_ch, in_ch, kh, kw)
            # Compute L1 norm for each output filter
            filter_l1 = torch.sum(torch.abs(weights), dim=(1, 2, 3))
            importance_scores[layer_name] = filter_l1.cpu().numpy()

    return importance_scores


def create_pruned_model_pytorch(original_model, prune_ratio, upsampling_factor, use_depth_wise_conv=False):
    """
    Create a new, smaller PyTorch model by pruning filters from the original.

    NOTE: F6 layer and upsample_blocks are NOT pruned to preserve the upsampling path.
    Only F1-F5 encoder/decoder layers are pruned.

    Args:
        original_model: Trained PyTorch model
        prune_ratio: Fraction of filters to remove
        upsampling_factor: Upsampling factor
        use_depth_wise_conv: Whether the model uses depthwise separable convolutions

    Returns:
        pruned_model: New smaller PyTorch model with transferred weights
        architecture_info: Dict with channel counts for each layer
    """
    print("\n" + "=" * 70)
    print("CREATING PRUNED MODEL")
    print("=" * 70)
    print(f"Prune ratio: {prune_ratio} ({prune_ratio * 100:.1f}% of filters removed)")
    print(f"Architecture type: {'Depthwise Separable' if use_depth_wise_conv else 'Standard Conv'}")
    print(f"NOTE: F6 and upsample_blocks are NOT pruned (kept at original size)")

    device = next(original_model.parameters()).device

    # Compute filter importance scores (F6 excluded)
    importance_scores = compute_filter_importance(original_model, use_depth_wise_conv=use_depth_wise_conv)

    # Determine architecture after pruning
    # Only prune F1-F5, NOT F6 (to keep upsample_blocks unchanged)
    layer_names_to_prune = ['conv_bn_relu1', 'conv_bn_relu2', 'conv_bn_relu3',
                            'conv_bn_relu4', 'conv_bn_relu5']
    original_channels = {'F1': 32, 'F2': 64, 'F3': 128, 'F4': 256, 'F5': 128, 'F6': 64}
    layer_to_F = {'conv_bn_relu1': 'F1', 'conv_bn_relu2': 'F2', 'conv_bn_relu3': 'F3',
                  'conv_bn_relu4': 'F4', 'conv_bn_relu5': 'F5', 'conv_bn_relu6': 'F6'}

    pruned_channels = {}
    keep_indices = {}

    # Prune F1-F5
    for layer_name in layer_names_to_prune:
        F_name = layer_to_F[layer_name]
        orig_filters = original_channels[F_name]
        num_keep = max(int(orig_filters * (1 - prune_ratio)), 4)

        if layer_name in importance_scores:
            scores = importance_scores[layer_name]
            # Get indices of top-k most important filters
            indices = np.argsort(scores)[-num_keep:]
            indices = np.sort(indices)  # Keep original order
            keep_indices[layer_name] = indices
        else:
            keep_indices[layer_name] = np.arange(num_keep)

        pruned_channels[F_name] = num_keep
        print(f"  {F_name}: {orig_filters} -> {num_keep} filters")

    # F6 is NOT pruned - keep all 64 channels
    pruned_channels['F6'] = 64
    keep_indices['conv_bn_relu6'] = np.arange(64)  # Keep all channels
    print(f"  F6: 64 -> 64 filters (NOT PRUNED - preserves upsample_blocks)")

    # Build new smaller model
    pruned_model = CNNUpsample(
        in_channels=1,
        upsampling_factor=upsampling_factor,
        pruned_channels=pruned_channels,
        use_depth_wise_conv=use_depth_wise_conv
    ).to(device)

    # Transfer weights from original to pruned model
    print("\nTransferring weights...")
    if use_depth_wise_conv:
        _transfer_pruned_weights_depthwise(original_model, pruned_model, keep_indices,
                                           layer_names_to_prune + ['conv_bn_relu6'])
    else:
        _transfer_pruned_weights_pytorch(original_model, pruned_model, keep_indices,
                                         layer_names_to_prune + ['conv_bn_relu6'])

    # Create architecture info for saving
    architecture_info = {
        'pruned': True,
        'prune_ratio': prune_ratio,
        'channels': pruned_channels,
        'upsampling_factor': upsampling_factor,
        'use_depth_wise_conv': use_depth_wise_conv,
        'keep_indices': {k: v.tolist() for k, v in keep_indices.items()},
        'upsample_blocks_unchanged': True  # Flag indicating upsample_blocks are not modified
    }

    # Validation
    orig_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction = (1 - pruned_params / orig_params) * 100

    print(f"\nOriginal parameters: {orig_params:,}")
    print(f"Pruned parameters: {pruned_params:,}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Note: upsample_blocks preserved at original size")
    print("=" * 70)

    return pruned_model, architecture_info


def _transfer_pruned_weights_pytorch(original_model, pruned_model, keep_indices, layer_names):
    """
    Transfer weights from original model to pruned model (standard convolution)

    NOTE: F6 is not pruned, so upsample_blocks weights are copied directly without modification.
    """

    prev_keep = None

    for i, layer_name in enumerate(layer_names):
        orig_block = getattr(original_model, layer_name)
        pruned_block = getattr(pruned_model, layer_name)

        curr_keep = keep_indices[layer_name]

        # Get original weights
        orig_conv_weight = orig_block.conv.weight.data  # (out, in, kh, kw)
        orig_bn_weight = orig_block.bn.weight.data
        orig_bn_bias = orig_block.bn.bias.data
        orig_bn_mean = orig_block.bn.running_mean.data
        orig_bn_var = orig_block.bn.running_var.data

        # Prune output channels
        pruned_conv_weight = orig_conv_weight[curr_keep]

        # Prune input channels (if not first layer)
        if prev_keep is not None:
            pruned_conv_weight = pruned_conv_weight[:, prev_keep]

        # Set pruned weights
        pruned_block.conv.weight.data = pruned_conv_weight

        # Handle conv bias (only if bias exists)
        if orig_block.conv.bias is not None:
            pruned_block.conv.bias.data = orig_block.conv.bias.data[curr_keep]

        pruned_block.bn.weight.data = orig_bn_weight[curr_keep]
        pruned_block.bn.bias.data = orig_bn_bias[curr_keep]
        pruned_block.bn.running_mean.data = orig_bn_mean[curr_keep]
        pruned_block.bn.running_var.data = orig_bn_var[curr_keep]

        prev_keep = curr_keep

    # Transfer upsampling block weights - these are NOT pruned
    # Since F6 keeps all 64 channels, upsample_blocks input doesn't change
    for idx, (orig_block, pruned_block) in enumerate(zip(original_model.upsample_blocks,
                                                         pruned_model.upsample_blocks)):
        orig_conv = orig_block['conv_bn_relu'].conv
        orig_bn = orig_block['conv_bn_relu'].bn
        pruned_conv = pruned_block['conv_bn_relu'].conv
        pruned_bn = pruned_block['conv_bn_relu'].bn

        # Copy all weights directly - no pruning for upsample_blocks
        pruned_conv.weight.data = orig_conv.weight.data.clone()

        # Copy bias weights (only if bias exists)
        if orig_conv.bias is not None:
            pruned_conv.bias.data = orig_conv.bias.data.clone()

        # Copy BN weights
        pruned_bn.weight.data = orig_bn.weight.data.clone()
        pruned_bn.bias.data = orig_bn.bias.data.clone()
        pruned_bn.running_mean.data = orig_bn.running_mean.data.clone()
        pruned_bn.running_var.data = orig_bn.running_var.data.clone()

    # Transfer prediction layer
    pruned_model.prediction.weight.data = original_model.prediction.weight.data.clone()


def _transfer_pruned_weights_depthwise(original_model, pruned_model, keep_indices, layer_names):
    """
    Transfer weights from original model to pruned model (depthwise separable convolution)

    NOTE: F6 is not pruned, and upsample_blocks always use standard ConvBNReLU (not depthwise).
    Since F6 keeps all 64 channels, upsample_blocks are copied directly without modification.
    """

    prev_keep = None

    for i, layer_name in enumerate(layer_names):
        orig_block = getattr(original_model, layer_name)
        pruned_block = getattr(pruned_model, layer_name)

        curr_keep = keep_indices[layer_name]

        # === Depthwise convolution weights ===
        # Depthwise conv: (in_ch, 1, kh, kw) - operates on each channel independently
        orig_dw_weight = orig_block.depthwise.weight.data
        orig_bn1_weight = orig_block.bn1.weight.data
        orig_bn1_bias = orig_block.bn1.bias.data
        orig_bn1_mean = orig_block.bn1.running_mean.data
        orig_bn1_var = orig_block.bn1.running_var.data

        # === Pointwise convolution weights ===
        # Pointwise conv: (out_ch, in_ch, 1, 1)
        orig_pw_weight = orig_block.pointwise.weight.data
        orig_bn2_weight = orig_block.bn2.weight.data
        orig_bn2_bias = orig_block.bn2.bias.data
        orig_bn2_mean = orig_block.bn2.running_mean.data
        orig_bn2_var = orig_block.bn2.running_var.data

        # For depthwise separable:
        # - Depthwise input channels = previous layer output channels
        # - Depthwise output channels = depthwise input channels (groups=in_channels)
        # - Pointwise input channels = depthwise output channels
        # - Pointwise output channels = this layer's output (curr_keep)

        if prev_keep is not None:
            # Prune depthwise based on previous layer's output channels
            pruned_dw_weight = orig_dw_weight[prev_keep]
            pruned_bn1_weight = orig_bn1_weight[prev_keep]
            pruned_bn1_bias = orig_bn1_bias[prev_keep]
            pruned_bn1_mean = orig_bn1_mean[prev_keep]
            pruned_bn1_var = orig_bn1_var[prev_keep]

            # Prune pointwise: output channels = curr_keep, input channels = prev_keep
            pruned_pw_weight = orig_pw_weight[curr_keep][:, prev_keep]
        else:
            # First layer: no input channel pruning for depthwise
            pruned_dw_weight = orig_dw_weight
            pruned_bn1_weight = orig_bn1_weight
            pruned_bn1_bias = orig_bn1_bias
            pruned_bn1_mean = orig_bn1_mean
            pruned_bn1_var = orig_bn1_var

            # Prune pointwise: only output channels
            pruned_pw_weight = orig_pw_weight[curr_keep]

        # Set depthwise weights
        pruned_block.depthwise.weight.data = pruned_dw_weight
        # Handle depthwise bias (only if bias exists)
        if orig_block.depthwise.bias is not None:
            if prev_keep is not None:
                pruned_block.depthwise.bias.data = orig_block.depthwise.bias.data[prev_keep]
            else:
                pruned_block.depthwise.bias.data = orig_block.depthwise.bias.data.clone()

        pruned_block.bn1.weight.data = pruned_bn1_weight
        pruned_block.bn1.bias.data = pruned_bn1_bias
        pruned_block.bn1.running_mean.data = pruned_bn1_mean
        pruned_block.bn1.running_var.data = pruned_bn1_var

        # Set pointwise weights
        pruned_block.pointwise.weight.data = pruned_pw_weight
        # Handle pointwise bias (only if bias exists)
        if orig_block.pointwise.bias is not None:
            pruned_block.pointwise.bias.data = orig_block.pointwise.bias.data[curr_keep]

        pruned_block.bn2.weight.data = orig_bn2_weight[curr_keep]
        pruned_block.bn2.bias.data = orig_bn2_bias[curr_keep]
        pruned_block.bn2.running_mean.data = orig_bn2_mean[curr_keep]
        pruned_block.bn2.running_var.data = orig_bn2_var[curr_keep]

        prev_keep = curr_keep

    # Transfer upsampling block weights - these use standard ConvBNReLU (not depthwise)
    # Since F6 keeps all 64 channels, copy weights directly without modification
    for idx, (orig_block, pruned_block) in enumerate(zip(original_model.upsample_blocks,
                                                         pruned_model.upsample_blocks)):
        orig_conv = orig_block['conv_bn_relu'].conv
        orig_bn = orig_block['conv_bn_relu'].bn
        pruned_conv = pruned_block['conv_bn_relu'].conv
        pruned_bn = pruned_block['conv_bn_relu'].bn

        # Copy all weights directly - no pruning for upsample_blocks
        pruned_conv.weight.data = orig_conv.weight.data.clone()

        # Copy bias weights (only if bias exists)
        if orig_conv.bias is not None:
            pruned_conv.bias.data = orig_conv.bias.data.clone()

        # Copy BN weights
        pruned_bn.weight.data = orig_bn.weight.data.clone()
        pruned_bn.bias.data = orig_bn.bias.data.clone()
        pruned_bn.running_mean.data = orig_bn.running_mean.data.clone()
        pruned_bn.running_var.data = orig_bn.running_var.data.clone()

    # Transfer prediction layer
    pruned_model.prediction.weight.data = original_model.prediction.weight.data.clone()


# ============================================================================
# Weight Conversion Functions (Standard <-> Depthwise)
# ============================================================================

def convert_standard_to_depthwise(standard_model, upsampling_factor):
    """
    Convert a standard convolution model to depthwise separable convolution.

    This creates a new depthwise model and attempts to initialize it with
    approximated weights from the standard model using SVD decomposition.

    Note: This is an approximation and may require fine-tuning for best results.

    Args:
        standard_model: Standard CNNUpsample model
        upsampling_factor: Upsampling factor

    Returns:
        depthwise_model: New CNNUpsample model with depthwise separable convolutions
    """
    print("\n" + "=" * 70)
    print("CONVERTING STANDARD MODEL TO DEPTHWISE SEPARABLE")
    print("=" * 70)

    device = next(standard_model.parameters()).device

    # Get channel counts from original model
    pruned_channels = None
    if standard_model.is_pruned:
        pruned_channels = {
            'F1': standard_model.conv_bn_relu1.conv.out_channels,
            'F2': standard_model.conv_bn_relu2.conv.out_channels,
            'F3': standard_model.conv_bn_relu3.conv.out_channels,
            'F4': standard_model.conv_bn_relu4.conv.out_channels,
            'F5': standard_model.conv_bn_relu5.conv.out_channels,
            'F6': standard_model.conv_bn_relu6.conv.out_channels,
        }

    # Create depthwise model with same architecture
    depthwise_model = CNNUpsample(
        in_channels=1,
        upsampling_factor=upsampling_factor,
        pruned_channels=pruned_channels,
        use_depth_wise_conv=True
    ).to(device)

    # Transfer and approximate weights
    layer_names = ['conv_bn_relu1', 'conv_bn_relu2', 'conv_bn_relu3',
                   'conv_bn_relu4', 'conv_bn_relu5', 'conv_bn_relu6']

    for layer_name in layer_names:
        std_block = getattr(standard_model, layer_name)
        dw_block = getattr(depthwise_model, layer_name)

        # Get standard conv weights: (out_ch, in_ch, kh, kw)
        std_weight = std_block.conv.weight.data

        # Approximate with depthwise separable using simple initialization
        # Depthwise: learn spatial patterns (in_ch, 1, kh, kw)
        # Pointwise: learn channel mixing (out_ch, in_ch, 1, 1)
        out_ch, in_ch, kh, kw = std_weight.shape

        # Initialize depthwise with average spatial pattern per input channel
        dw_weight = std_weight.mean(dim=0, keepdim=True).transpose(0, 1)  # (in_ch, 1, kh, kw)
        dw_block.depthwise.weight.data = dw_weight
        # Initialize depthwise bias only if the target model has bias
        if dw_block.depthwise.bias is not None:
            dw_block.depthwise.bias.data = torch.zeros(in_ch, device=device)

        # Initialize pointwise to approximate channel mixing
        # Use average across spatial dimensions
        pw_weight = std_weight.mean(dim=(2, 3), keepdim=True)  # (out_ch, in_ch, 1, 1)
        dw_block.pointwise.weight.data = pw_weight
        # Transfer bias to pointwise only if both source and target have bias
        if dw_block.pointwise.bias is not None:
            if std_block.conv.bias is not None:
                dw_block.pointwise.bias.data = std_block.conv.bias.data.clone()
            else:
                dw_block.pointwise.bias.data = torch.zeros(out_ch, device=device)

        # Copy BN parameters
        dw_block.bn1.weight.data = torch.ones(in_ch, device=device)
        dw_block.bn1.bias.data = torch.zeros(in_ch, device=device)
        dw_block.bn1.running_mean.data = torch.zeros(in_ch, device=device)
        dw_block.bn1.running_var.data = torch.ones(in_ch, device=device)

        dw_block.bn2.weight.data = std_block.bn.weight.data.clone()
        dw_block.bn2.bias.data = std_block.bn.bias.data.clone()
        dw_block.bn2.running_mean.data = std_block.bn.running_mean.data.clone()
        dw_block.bn2.running_var.data = std_block.bn.running_var.data.clone()

    # Transfer upsample block weights
    for idx, (std_block, dw_block) in enumerate(zip(standard_model.upsample_blocks,
                                                    depthwise_model.upsample_blocks)):
        std_conv = std_block['conv_bn_relu'].conv
        std_bn = std_block['conv_bn_relu'].bn

        dw_conv = dw_block['conv_bn_relu']
        std_weight = std_conv.weight.data
        out_ch, in_ch, kh, kw = std_weight.shape

        # Initialize depthwise
        dw_weight = std_weight.mean(dim=0, keepdim=True).transpose(0, 1)
        dw_conv.depthwise.weight.data = dw_weight
        if dw_conv.depthwise.bias is not None:
            dw_conv.depthwise.bias.data = torch.zeros(in_ch, device=device)

        # Initialize pointwise
        pw_weight = std_weight.mean(dim=(2, 3), keepdim=True)
        dw_conv.pointwise.weight.data = pw_weight
        if dw_conv.pointwise.bias is not None:
            if std_conv.bias is not None:
                dw_conv.pointwise.bias.data = std_conv.bias.data.clone()
            else:
                dw_conv.pointwise.bias.data = torch.zeros(out_ch, device=device)

        # Copy BN parameters
        dw_conv.bn1.weight.data = torch.ones(in_ch, device=device)
        dw_conv.bn1.bias.data = torch.zeros(in_ch, device=device)
        dw_conv.bn1.running_mean.data = torch.zeros(in_ch, device=device)
        dw_conv.bn1.running_var.data = torch.ones(in_ch, device=device)

        dw_conv.bn2.weight.data = std_bn.weight.data.clone()
        dw_conv.bn2.bias.data = std_bn.bias.data.clone()
        dw_conv.bn2.running_mean.data = std_bn.running_mean.data.clone()
        dw_conv.bn2.running_var.data = std_bn.running_var.data.clone()

    # Transfer prediction layer
    depthwise_model.prediction.weight.data = standard_model.prediction.weight.data.clone()

    # Print parameter comparison
    std_params = sum(p.numel() for p in standard_model.parameters())
    dw_params = sum(p.numel() for p in depthwise_model.parameters())
    reduction = (1 - dw_params / std_params) * 100

    print(f"Standard model parameters: {std_params:,}")
    print(f"Depthwise model parameters: {dw_params:,}")
    print(f"Parameter reduction: {reduction:.1f}%")
    print("\nNote: Weight approximation used. Fine-tuning recommended for best results.")
    print("=" * 70)

    return depthwise_model


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(patches, heatmaps, model_path, epochs, batch_size, val_split=0.3,
                lr=0.001, pretrained_path='', upsampling_factor=2,
                use_pruning=False, prune_ratio=0.3, sparsity_lambda=1e-5,
                use_depth_wise_conv=False,
                pixel_size=107, wavelength=715, numerical_aperture=1.49,
                L2_weighting_factor=100):
    """
    Train model with optional pruning and depthwise separable convolution support.

    If use_pruning=True:
    1. Trains with L1 regularization to encourage sparse filters
    2. After training, creates a physically smaller pruned model
    3. Saves both PyTorch weights and architecture info JSON

    If use_depth_wise_conv=True:
    1. Uses depthwise separable convolutions for faster training and inference
    2. Significantly reduces parameter count (~8-9x per conv layer)
    3. Can be combined with pruning for maximum efficiency

    Args:
        patches: Training input patches (N, H, W)
        heatmaps: Training target heatmaps (N, H*up, W*up)
        model_path: Path to save model
        epochs: Number of training epochs
        batch_size: Batch size
        val_split: Validation split ratio
        lr: Learning rate
        pretrained_path: Path to pretrained weights (optional)
        upsampling_factor: Upsampling factor
        use_pruning: Enable post-training pruning
        prune_ratio: Fraction of filters to prune
        sparsity_lambda: L1 regularization strength for pruning
        use_depth_wise_conv: Use depthwise separable convolutions
        pixel_size: Pixel size in nm
        wavelength: Wavelength in nm
        numerical_aperture: Numerical aperture
        L2_weighting_factor: L2 weighting factor for loss

    Returns:
        history: Training history object
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        patches, heatmaps, test_size=val_split, random_state=42
    )

    # Convert to tensors and add channel dimension
    X_train = torch.from_numpy(X_train).float().unsqueeze(1)  # (N, 1, H, W)
    X_val = torch.from_numpy(X_val).float().unsqueeze(1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    y_val = torch.from_numpy(y_val).float().unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Create model
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        # Check if pretrained model has architecture info
        arch_path = os.path.join(os.path.dirname(pretrained_path), 'architecture_info.json')
        if os.path.exists(arch_path):
            with open(arch_path, 'r') as f:
                arch_info = json.load(f)
            pretrained_depthwise = arch_info.get('use_depth_wise_conv', False)
            pretrained_channels = arch_info.get('channels', None)

            # Warning if architecture mismatch
            if pretrained_depthwise != use_depth_wise_conv:
                print(f"Warning: Pretrained model uses {'depthwise' if pretrained_depthwise else 'standard'} "
                      f"convolutions, but training with {'depthwise' if use_depth_wise_conv else 'standard'}.")
                print("Creating new model with requested architecture...")
                model = CNNUpsample(in_channels=1, upsampling_factor=upsampling_factor,
                                    use_depth_wise_conv=use_depth_wise_conv).to(device)
            else:
                model = CNNUpsample(in_channels=1, upsampling_factor=upsampling_factor,
                                    pruned_channels=pretrained_channels,
                                    use_depth_wise_conv=use_depth_wise_conv).to(device)
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            # No architecture info, assume standard model
            model = CNNUpsample(in_channels=1, upsampling_factor=upsampling_factor,
                                use_depth_wise_conv=use_depth_wise_conv).to(device)
            if not use_depth_wise_conv:
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
            else:
                print("Warning: Loading standard weights into depthwise model not supported.")
                print("Training from scratch with depthwise architecture.")
    else:
        model = CNNUpsample(in_channels=1, upsampling_factor=upsampling_factor,
                            use_depth_wise_conv=use_depth_wise_conv).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel architecture: {'Depthwise Separable' if use_depth_wise_conv else 'Standard Conv'}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with optional sparsity regularization
    criterion = CustomLoss(sparsity_lambda=sparsity_lambda if use_pruning else 0.0).to(device)

    if use_pruning:
        print(f"\n*** Training with sparsity regularization (lambda={sparsity_lambda}) ***")

    if use_depth_wise_conv:
        print(f"\n*** Training with depthwise separable convolutions ***")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

    # Callbacks
    # os.makedirs(model_path, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_path, "best_weights.pth"),
        save_best_only=True, verbose=True
    )
    history = LossHistory()

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y, model if use_pruning else None)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update history
        history.append(train_loss, val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Model checkpoint
        checkpoint(model, val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - lr: {optimizer.param_groups[0]['lr']:.2e}")

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(model_path, "best_weights.pth"), map_location=device))

    # Apply pruning if enabled
    if use_pruning:
        print("\n*** Applying post-training pruning ***")
        pruned_model, architecture_info = create_pruned_model_pytorch(
            model, prune_ratio, upsampling_factor, use_depth_wise_conv=use_depth_wise_conv
        )
        torch.save(pruned_model.state_dict(), os.path.join(model_path, 'best_weights_prune.pth'))
        print(f"✓ Saved pruned PyTorch weights: best_weights_prune.pth")

        # Save architecture info JSON
        arch_path = os.path.join(model_path, 'architecture_info.json')
        with open(arch_path, 'w') as f:
            json.dump(architecture_info, f, indent=2)
        print(f"✓ Saved architecture info: architecture_info.json")

        # Keep the original full model too
        torch.save(model.state_dict(), os.path.join(model_path, 'final_model.pth'))
    else:
        # Standard saving (no pruning)
        torch.save(model.state_dict(), os.path.join(model_path, 'best_weights.pth'))
        torch.save(model.state_dict(), os.path.join(model_path, 'final_model.pth'))

        # Save architecture info for depthwise models even without pruning
        if use_depth_wise_conv:
            architecture_info = {
                'pruned': False,
                'upsampling_factor': upsampling_factor,
                'use_depth_wise_conv': True,
                'channels': model.get_channel_counts()
            }
            arch_path = os.path.join(model_path, 'architecture_info.json')
            with open(arch_path, 'w') as f:
                json.dump(architecture_info, f, indent=2)
            print(f"✓ Saved architecture info: architecture_info.json")

    # Save model metadata
    mdict = {
        "upsampling_factor": upsampling_factor,
        "Normalization factor": L2_weighting_factor,
        "pixel_size": pixel_size,
        "wavelength": wavelength,
        "numerical_aperture": numerical_aperture,
        "use_depth_wise_conv": int(use_depth_wise_conv)  # Save as int for MATLAB compatibility
    }
    sio.savemat(os.path.join(model_path, "model_metadata.mat"), mdict)

    return history


# ============================================================================
# Data Generation Functions
# ============================================================================

def FromLoc2Image_SimpleHistogram(xc_array, yc_array, image_size=(64, 64), pixel_size=100):
    w = image_size[0]
    h = image_size[1]
    locImage = np.zeros((image_size[0], image_size[1]))
    n_locs = len(xc_array)
    for e in range(n_locs):
        locImage[int(max(min(round(yc_array[e] / pixel_size), w - 1), 0))][
            int(max(min(round(xc_array[e] / pixel_size), h - 1), 0))] += 1
    return locImage


def FromLoc2Image_Erf(xc_array, yc_array, photon_array, sigma_array, image_size=(64, 64), pixel_size=100):
    w = image_size[0]
    h = image_size[1]
    erfImage = np.zeros((w, h))
    for ij in range(w * h):
        j = int(ij / w)
        i = ij - j * w
        for (xc, yc, photon, sigma) in zip(xc_array, yc_array, photon_array, sigma_array):
            # Don't bother if the emitter has photons <= 0 or if Sigma <= 0
            if (sigma > 0) and (photon > 0):
                S = sigma * math.sqrt(2)
                x = i * pixel_size - xc
                y = j * pixel_size - yc
                # Don't bother if the emitter is further than 4 sigma from the centre of the pixel
                if (x + pixel_size / 2) ** 2 + (y + pixel_size / 2) ** 2 < 16 * sigma ** 2:
                    ErfX = math.erf((x + pixel_size) / S) - math.erf(x / S)
                    ErfY = math.erf((y + pixel_size) / S) - math.erf(y / S)
                    erfImage[j][i] += 0.25 * photon * ErfX * ErfY
    return erfImage


# ============================================================================
# Report Generation
# ============================================================================

def generate_training_report(model_path, simParameters, history):
    """Generate training report with loss curves"""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Training Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Model path: {model_path}", ln=True)
        pdf.ln(5)
        for key, value in simParameters.items():
            pdf.cell(200, 10, f"{key}: {value}", ln=True)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Loss Curve", ln=True)
        pdf.ln(5)

        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = os.path.join(model_path, "loss_curve.png")
        plt.savefig(loss_plot_path)
        plt.close()

        pdf.image(loss_plot_path, x=10, y=None, w=180)
        pdf_output_path = os.path.join(model_path, "training_report.pdf")
        pdf.output(pdf_output_path)
        print(f"Training report saved at {pdf_output_path}")
    except ImportError:
        print("fpdf not installed, skipping PDF report generation")
        # Still save the loss curve
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model_path, "loss_curve.png"))
        plt.close()


# ============================================================================
# Normalization Functions
# ============================================================================

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x


# ============================================================================
# Main Script - Example Usage
# ============================================================================

if __name__ == "__main__":
    print('=' * 70)
    print('AutoDS PyTorch Training')
    print('=' * 70)

    print(f'Pruning mode: {"ENABLED" if config.use_pruning else "DISABLED"}')
    if config.use_pruning:
        print(f'Prune ratio: {config.prune_ratio} ({config.prune_ratio * 100:.0f}%)')
    print(f'Depthwise mode: {"ENABLED" if config.use_depth_wise_conv else "DISABLED"}')
    print('=' * 70)

    # Loop through all diff versions
    for diff_version in ['diff_1', 'diff_2', 'diff_3', 'diff_4']:
        print('Starting training for {diff_version}')
        # Update config paths for this version
        config.tf_model_metadata_path = f'../models/v0-tensorflow/{diff_version}/model_metadata.mat'
        config.tf_training_report_path = f'../models/v0-tensorflow/{diff_version}/training_report.pdf'
        config.model_name = f'{diff_version}'

        load_metadata_from_files()
        print_current_config()

        # Save results
        os.makedirs(config.model_save_path, exist_ok=True)
        # ====================================================================
        # from re import L

        # ---------------------------- User input ----------------------------
        # @markdown #2. Generate simulated dataset
        # @markdown ---
        # @markdown Camera settings:
        FOV_size = config.FOV_size  # @param {type:"number"}
        pixel_size = config.pixel_size  # @param {type:"number"}
        ADC_per_photon_conversion = config.ADC_per_photon_conversion  # @param {type:"number"}
        ReadOutNoise_ADC = config.ReadOutNoise_ADC  # @param {type:"number"}
        ADC_offset = config.ADC_offset  # @param {type:"number"}

        # @markdown Acquisition settings:
        emitter_density = config.emitter_density  # @param {type:"number"}
        emitter_density_std = config.emitter_density_std  # @param {type:"number"}

        number_of_frames = config.number_of_frames  # @param {type:"integer"}

        wavelength = config.wavelength  # @param {type:"number"}
        numerical_aperture = config.numerical_aperture  # @param {type:"number"}

        sigma_std = config.sigma_std  # @param {type:"number"}
        n_photons = config.n_photons  # @param {type:"number"}
        n_photons_std = config.n_photons_std  # @param {type:"number"}

        # ---------------------------- Variable initialisation ----------------------------
        # Start the clock to measure how long it takes
        start = time.time()

        print('-----------------------------------------------------------')
        n_molecules = emitter_density * FOV_size * FOV_size / 10 ** 6
        n_molecules_std = emitter_density_std * FOV_size * FOV_size / 10 ** 6
        print('Number of molecules / FOV: ' + str(round(n_molecules, 2)) + ' +/- ' + str((round(n_molecules_std, 2))))

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ---> sigma = FWHM / 2.355 = 0.61 * wavelength / 2.355 * NA
        sigma = 0.26 * wavelength / numerical_aperture
        print('Gaussian PSF sigma: ' + str(round(sigma, 2)) + ' nm')

        M = round(FOV_size / pixel_size)
        N = round(FOV_size / pixel_size)

        FOV_size = M * pixel_size
        print('Final image size: ' + str(M) + 'x' + str(M) + ' (' + str(round(FOV_size / 1000, 3)) + 'um x' + str(
            round(FOV_size / 1000, 3)) + ' um)')

        np.random.seed(1)
        display_upsampling = 8  # used to display the loc map here
        NoiseFreeImages = np.zeros((number_of_frames, M, M))
        locImage = np.zeros((number_of_frames, display_upsampling * M, display_upsampling * N))

        frames = []
        all_xloc = []
        all_yloc = []
        all_photons = []
        all_sigmas = []

        # ---------------------------- Main simulation loop ----------------------------
        print('-----------------------------------------------------------')
        for f in tqdm(range(number_of_frames)):
            # Define the coordinates of emitters by randomly distributing them across the FOV
            n_mol = int(max(round(np.random.normal(n_molecules, n_molecules_std, size=1)[0]), 0))
            x_c = np.random.uniform(low=0.0, high=FOV_size, size=n_mol)
            y_c = np.random.uniform(low=0.0, high=FOV_size, size=n_mol)
            photon_array = np.random.normal(n_photons, n_photons_std, size=n_mol)
            sigma_array = np.random.normal(sigma, sigma_std, size=n_mol)

            all_xloc += x_c.tolist()
            all_yloc += y_c.tolist()
            frames += ((f + 1) * np.ones(x_c.shape[0])).tolist()
            all_photons += photon_array.tolist()
            all_sigmas += sigma_array.tolist()

            locImage[f] = FromLoc2Image_SimpleHistogram(x_c, y_c,
                                                        image_size=(N * display_upsampling, M * display_upsampling),
                                                        pixel_size=pixel_size / display_upsampling)
            NoiseFreeImages[f] = FromLoc2Image_Erf(x_c, y_c, photon_array, sigma_array, image_size=(M, M),
                                                   pixel_size=pixel_size)

        # ---------------------------- Create DataFrame fof localization file ----------------------------
        # Table with localization info as dataframe output
        LocData = pd.DataFrame()
        LocData["frame"] = frames
        LocData["x [nm]"] = all_xloc
        LocData["y [nm]"] = all_yloc
        LocData["Photon #"] = all_photons
        LocData["Sigma [nm]"] = all_sigmas
        LocData.index += 1  # set indices to start at 1 and not 0 (same as ThunderSTORM)

        # ---------------------------- Estimation of SNR ----------------------------
        n_frames_for_SNR = 100
        M_SNR = 10
        x_c = np.random.uniform(low=0.0, high=pixel_size * M_SNR, size=n_frames_for_SNR)
        y_c = np.random.uniform(low=0.0, high=pixel_size * M_SNR, size=n_frames_for_SNR)
        photon_array = np.random.normal(n_photons, n_photons_std, size=n_frames_for_SNR)
        sigma_array = np.random.normal(sigma, sigma_std, size=n_frames_for_SNR)

        SNR = np.zeros(n_frames_for_SNR)
        for i in range(n_frames_for_SNR):
            SingleEmitterImage = FromLoc2Image_Erf(np.array([x_c[i]]), np.array([x_c[i]]), np.array([photon_array[i]]),
                                                   np.array([sigma_array[i]]), (M_SNR, M_SNR), pixel_size)
            Signal_photon = np.max(SingleEmitterImage)
            Noise_photon = math.sqrt((ReadOutNoise_ADC / ADC_per_photon_conversion) ** 2 + Signal_photon)
            SNR[i] = Signal_photon / Noise_photon

        print('SNR: ' + str(round(np.mean(SNR), 2)) + ' +/- ' + str(round(np.std(SNR), 2)))
        # ---------------------------- ----------------------------

        # Table with info
        savedParameters = {}
        savedParameters["FOV size (nm)"] = FOV_size
        savedParameters["Pixel size (nm)"] = pixel_size
        savedParameters["ADC/photon"] = ADC_per_photon_conversion
        savedParameters["Read-out noise (ADC)"] = ReadOutNoise_ADC
        savedParameters["Constant offset (ADC)"] = ADC_offset

        savedParameters["Emitter density (emitters/um^2)"] = emitter_density
        savedParameters["STD of emitter density (emitters/um^2)"] = emitter_density_std
        savedParameters["Number of frames"] = number_of_frames

        savedParameters["Wavelength (nm)"] = wavelength
        savedParameters["NA"] = numerical_aperture
        savedParameters["Sigma (nm)"] = sigma
        savedParameters["STD of Sigma (nm)"] = sigma_std
        savedParameters["Number of photons"] = n_photons
        savedParameters["STD of number of photons"] = n_photons_std
        savedParameters["SNR"] = np.mean(SNR)
        savedParameters["STD of SNR"] = np.std(SNR)

        # ---------------------------- Finish simulation ----------------------------
        # Calculating the noisy image
        Images = ADC_per_photon_conversion * np.random.poisson(NoiseFreeImages) + ReadOutNoise_ADC * np.random.normal(
            size=(number_of_frames, M, N)) + ADC_offset
        Images[Images <= 0] = 0

        # Convert to 16-bit or 32-bits integers
        if Images.max() < (2 ** 16 - 1):
            Images = Images.astype(np.uint16)
        else:
            Images = Images.astype(np.uint32)

        # ---------------------------- Display ----------------------------
        # Displaying the time elapsed for simulation
        dt = time.time() - start
        minutes, seconds = divmod(dt, 60)
        hours, minutes = divmod(minutes, 60)
        print("Time elapsed:", hours, "hour(s)", minutes, "min(s)", round(seconds, 1), "sec(s)")

        # -------------------- User input --------------------
        patch_size = config.patch_size  # @param {type:"integer"}
        upsampling_factor = 8  # @param ["4", "8", "16"] {type:"raw"}
        num_patches_per_frame = config.num_patches_per_frame  # @param {type:"integer"}
        min_number_of_emitters_per_patch = 0  # @param {type:"integer"}
        max_num_patches = 10000  # @param {type:"integer"}
        gaussian_sigma = 1  # @param {type:"integer"}

        savedParameters["patch size"] = patch_size
        savedParameters["upsampling factor"] = upsampling_factor
        savedParameters["num_patches_per_frame"] = num_patches_per_frame
        savedParameters["min_number_of_emitters_per_patch"] = min_number_of_emitters_per_patch
        savedParameters["max_num_patches"] = max_num_patches
        savedParameters["gaussian_sigma"] = gaussian_sigma

        # @markdown Estimate the optimal normalization factor automatically?
        Automatic_normalization = False  # @param {type:"boolean"}
        # @markdown Otherwise, it will use the following value:
        L2_weighting_factor = 100  # @param {type:"number"}

        # -------------------- Prepare variables --------------------
        # Start the clock to measure how long it takes
        start = time.time()

        # Initialize some parameters
        pixel_size_hr = pixel_size / int(upsampling_factor)  # in nm
        n_patches = min(int(number_of_frames) * int(num_patches_per_frame), int(max_num_patches))
        # patch_size = int(patch_size) * int(upsampling_factor)
        patch_size = int(config.patch_size)
        patch_size_hr = patch_size * upsampling_factor

        # Initialize the training patches and labels
        patches = np.zeros((n_patches, patch_size, patch_size), dtype=np.float32)
        spikes = np.zeros((n_patches, patch_size_hr, patch_size_hr), dtype=np.float32)
        heatmaps = np.zeros((n_patches, patch_size_hr, patch_size_hr), dtype=np.float32)

        # Run over all frames and construct the training examples
        k = 1  # current patch count
        skip_counter = 0  # number of dataset skipped due to low density
        id_start = 0  # id position in LocData for current frame
        print('Generating ' + str(n_patches) + ' patches of ' + str(patch_size) + 'x' + str(patch_size))

        n_locs = len(LocData.index)
        print('Total number of localizations: ' + str(n_locs))
        density = n_locs / (M * N * number_of_frames * (0.001 * pixel_size) ** 2)
        print('Density: ' + str(round(density, 2)) + ' locs/um^2')
        n_locs_per_patch = density * (patch_size ** 2)

        if Automatic_normalization:
            # This empirical formulae attempts to balance the loss L2 function between the background and the bright spikes
            # A value of 100 was originally chosen to balance L2 for a patch size of 2.6x2.6^2 0.1um pixel size and density of 3 (hence the 20.28), at upsampling_factor = 8
            L2_weighting_factor = 100 / math.sqrt(
                min(n_locs_per_patch, min_number_of_emitters_per_patch) * 8 ** 2 / (
                        int(upsampling_factor) ** 2 * 20.28))
            print('Normalization factor: ' + str(round(L2_weighting_factor, 2)))

        savedParameters["L2 weighting factor"] = L2_weighting_factor
        # -------------------- Patch generation loop --------------------
        print('-----------------------------------------------------------')
        for (f, thisFrame) in enumerate(tqdm(Images)):

            # Upsample the frame
            # upsampledFrame = np.kron(thisFrame, np.ones((int(upsampling_factor),int(upsampling_factor))))

            thisFrame -= np.min(thisFrame)
            thisFrame[thisFrame < 0] = 0

            normFrame = normalize_im(thisFrame, thisFrame.mean(), thisFrame.std())

            # Read all the provided high-resolution locations for current frame
            DataFrame = LocData[LocData['frame'] == f + 1].copy()

            Mhr = M * upsampling_factor
            Nhr = N * upsampling_factor

            # Get the approximated locations according to the high-res grid pixel size
            Chr_emitters = [int(max(min(round(DataFrame['x [nm]'][i] / pixel_size_hr), Nhr - 1), 0)) for i in
                            range(id_start + 1, id_start + 1 + len(DataFrame.index))]
            Rhr_emitters = [int(max(min(round(DataFrame['y [nm]'][i] / pixel_size_hr), Mhr - 1), 0)) for i in
                            range(id_start + 1, id_start + 1 + len(DataFrame.index))]
            id_start += len(DataFrame.index)

            # Build Localization image
            LocImage = np.zeros((Mhr, Nhr))
            LocImage[(Rhr_emitters, Chr_emitters)] = 1

            # Here, there's a choice between the original Gaussian (classification approach) and using the erf function
            HeatMapImage = L2_weighting_factor * gaussian_filter(LocImage, float(gaussian_sigma))

            # Generate random position for the top left corner of the patch
            xc = np.random.randint(0, M - patch_size, size=num_patches_per_frame)
            yc = np.random.randint(0, N - patch_size, size=num_patches_per_frame)

            xc_hr = xc * upsampling_factor
            yc_hr = yc * upsampling_factor

            for c in range(len(xc)):
                if LocImage[xc_hr[c]:xc_hr[c] + patch_size_hr, yc_hr[c]:yc_hr[
                                                                            c] + patch_size_hr].sum() < min_number_of_emitters_per_patch:
                    skip_counter += 1
                    continue

                else:
                    # Limit maximal number of training examples to 15k
                    if k > max_num_patches:
                        break
                    else:
                        # Assign the patches to the right part of the images
                        patches[k - 1] = normFrame[xc[c]:xc[c] + patch_size, yc[c]:yc[c] + patch_size]
                        spikes[k - 1] = LocImage[xc_hr[c]:xc_hr[c] + patch_size_hr, yc_hr[c]:yc_hr[c] + patch_size_hr]
                        heatmaps[k - 1] = HeatMapImage[
                            xc_hr[c]:xc_hr[c] + patch_size_hr, yc_hr[c]:yc_hr[c] + patch_size_hr]
                        k += 1  # increment current patch count

        # Remove the empty data
        patches = patches[:k - 1]
        spikes = spikes[:k - 1]
        heatmaps = heatmaps[:k - 1]
        n_patches = k - 1
        # ----------------- Visualization ------------------
        num_samples = np.min([10, patches.shape[0]])
        random_ind = np.random.randint(0, patches.shape[0], num_samples)
        plt.figure()
        plt.suptitle("Visualization of random patches")
        for ind in range(num_samples):
            plt.subplot(2, num_samples // 2, ind + 1)
            plt.imshow(patches[random_ind[ind]])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        # -------------------- Failsafe --------------------
        # Check if the size of the training set is smaller than 5k to notify user to simulate more images using ThunderSTORM
        if ((k - 1) < 5000):
            print('!! WARNING: Training set size is below 5K - Consider simulating more images in ThunderSTORM. !!')

        # -------------------- Displays --------------------
        print('Number of patches skipped due to low density: ' + str(skip_counter))
        print(str(n_patches) + ' patches were generated.')

        # Displaying the time elapsed for training
        dt = time.time() - start
        minutes, seconds = divmod(dt, 60)
        hours, minutes = divmod(minutes, 60)
        print("Time elapsed:", hours, "hour(s)", minutes, "min(s)", round(seconds), "sec(s)")

        # @markdown #3. Initialize training parameters

        model_path = config.model_save_path  # @param {type: "string"}
        model_name = config.model_name  # @param {type: "string"}
        # model_path = os.path.join(model_path, model_name)
        number_of_epochs = config.number_of_epochs  # @param {type:"integer"}
        batch_size = config.batch_size  # @param {type:"integer"}

        percentage_validation = 20  # @param {type:"number"}
        initial_learning_rate = 0.001  # @param {type:"number"}

        percentage_validation /= 100

        # Pretrained model path initialised here so next cell does not need to be run
        h5_file_path = ''
        Use_pretrained_model = False

        if not ('patches' in globals()):
            print('!! WARNING: No patches were found in memory currently. !!')

        Save_path = os.path.join(model_path, model_name)
        if os.path.exists(Save_path):
            print('The model folder already exists and will be overwritten.')

        print('-----------------------------')
        print('Training parameters set.')

        # @markdown ###Loading weights from a pre-trained network

        # @markdown If you chose to continue training an existing model, please specify the hd5 file path
        h5_file_path = ''  # @param {type: "string"}

        # Display info about the pretrained model to be loaded (or not)
        if Use_pretrained_model:
            print('Weights found in:')
            print(h5_file_path)
            print('will be loaded prior to training.')
        else:
            print('No pretrained network will be used.')

        # @markdown #4. Start training

        # Start the clock to measure how long it takes
        start = time.time()

        # here we check that no model with the same name already exist, if so delete
        if os.path.exists(Save_path):
            shutil.rmtree(Save_path)

        # Create the model folder!
        os.makedirs(Save_path)

        # Let's go !
        train_history = train_model(patches,
                                    heatmaps,
                                    Save_path,
                                    epochs=number_of_epochs,
                                    batch_size=batch_size,
                                    val_split=percentage_validation,
                                    lr=initial_learning_rate,
                                    pretrained_path=h5_file_path,
                                    upsampling_factor=upsampling_factor,
                                    use_pruning=config.use_pruning,
                                    prune_ratio=config.prune_ratio,
                                    sparsity_lambda=config.sparsity_lambda,
                                    use_depth_wise_conv=config.use_depth_wise_conv,
                                    pixel_size=pixel_size,
                                    wavelength=config.wavelength,
                                    numerical_aperture=config.numerical_aperture,
                                    L2_weighting_factor=config.L2_weighting_factor
                                    )

        dt = time.time() - start
        minutes, seconds = divmod(dt, 60)
        hours, minutes = divmod(minutes, 60)
        print("Time elapsed:", hours, "hour(s)", minutes, "min(s)", round(seconds), "sec(s)")

        # export pdf after training to update the existing document
        generate_training_report(Save_path, savedParameters, train_history)