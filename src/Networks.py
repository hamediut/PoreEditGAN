
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import legacy
import dnnlib
from collections import OrderedDict

###-------------------------------Encoder-----------------------------
# __all__ = ['StyleGANEncoderNet']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# RESOLUTION = G.img_resolution
RESOLUTION = 256
class StyleGANEncoderNet(nn.Module):
  """Defines the encoder network for StyleGAN inversion.
  Contains the implementation of encoder for StyleGAN inversion.

  For more details, please check the paper:
    https://arxiv.org/pdf/2004.00049.pdf

  NOTE: The encoder takes images with `RGB` color channels and range [-1, 1]
  as inputs, and encode the input images to W+ space of StyleGAN.
  """

  def __init__(self,
               resolution = RESOLUTION,
               w_space_dim=512,
               image_channels=1, 
               encoder_channels_base=64,
               encoder_channels_max=1024,
               use_wscale=False,
               use_bn=False):
    """Initializes the encoder with basic settings.

    Args:
      resolution: The resolution of the input image.
      w_space_dim: The dimension of the disentangled latent vectors, w.
        (default: 512)
      image_channels: Number of channels of the input image. (default: 3)
      encoder_channels_base: Base factor of the number of channels used in
        residual blocks of encoder. (default: 64)
      encoder_channels_max: Maximum number of channels used in residual blocks
        of encoder. (default: 1024)
      use_wscale: Whether to use `wscale` layer. (default: False)
      use_bn: Whether to use batch normalization layer. (default: False)

    Raises:
      ValueError: If the input `resolution` is not supported.
    """
    super().__init__()

    if resolution not in _RESOLUTIONS_ALLOWED:
      raise ValueError(f'Invalid resolution: {resolution}!\n'
                       f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

    self.init_res = _INIT_RES
    self.resolution = resolution
    self.w_space_dim = w_space_dim
    self.image_channels = image_channels
    self.encoder_channels_base = encoder_channels_base
    self.encoder_channels_max = encoder_channels_max
    self.use_wscale = use_wscale
    self.use_bn = use_bn
    # Blocks used in encoder.
    self.num_blocks = int(np.log2(resolution)) # for res  = 256 --> 8 blocks, for res = 512--> 9 blocks
    # Layers used in generator.
    self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2 
    # for res =  256 --> 14 layers, for res = 512, 16 layers
    in_channels = self.image_channels
    out_channels = self.encoder_channels_base
    for block_idx in range(self.num_blocks):
      if block_idx == 0:
        self.add_module(
            f'block{block_idx}',
            FirstBlock(in_channels=in_channels,
                       out_channels=out_channels,
                       use_wscale=self.use_wscale,
                       use_bn=self.use_bn))
# first block is a con layer followed by a leakyRELU activation function
      elif block_idx == self.num_blocks - 1:
        in_channels = in_channels * self.init_res * self.init_res
        out_channels = self.w_space_dim * 2 * block_idx
        self.add_module(
            f'block{block_idx}',
            LastBlock(in_channels=in_channels,
                      out_channels=out_channels,
                      use_wscale=True,
                      use_bn=self.use_bn))

      else:
        self.add_module(
            f'block{block_idx}',
            ResBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     use_wscale=self.use_wscale,
                     use_bn=self.use_bn))
      in_channels = out_channels
      out_channels = min(out_channels * 2, self.encoder_channels_max)

    self.downsample = AveragePoolingLayer()

  def forward(self, x):
    if x.ndim != 4 or x.shape[1:] != (
        self.image_channels, self.resolution, self.resolution):
      raise ValueError(f'The input image should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {x.shape} is received!')

    for block_idx in range(self.num_blocks):
      if 0 < block_idx < self.num_blocks - 1:
        x = self.downsample(x)
      x = self.__getattr__(f'block{block_idx}')(x)
    return x


class AveragePoolingLayer(nn.Module):
  """Implements the average pooling layer.

  Basically, this layer can be used to downsample feature maps from spatial
  domain.
  """

  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    ksize = [self.scale_factor, self.scale_factor]
    strides = [self.scale_factor, self.scale_factor]
    return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


class BatchNormLayer(nn.Module):
  """Implements batch normalization layer."""

  def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon=1e-5):
    """Initializes with basic settings.

    Args:
      channels: Number of channels of the input tensor.
      gamma: Whether the scale (weight) of the affine mapping is learnable.
      beta: Whether the center (bias) of the affine mapping is learnable.
      decay: Decay factor for moving average operations in this layer.
      epsilon: A value added to the denominator for numerical stability.
    """
    super().__init__()
    self.bn = nn.BatchNorm2d(num_features=channels,
                             affine=True,
                             track_running_stats=True,
                             momentum=1 - decay,
                             eps=epsilon)
    self.bn.weight.requires_grad = gamma
    self.bn.bias.requires_grad = beta

  def forward(self, x):
    return self.bn(x)


class WScaleLayer(nn.Module):
  """Implements the layer to scale weight variable and add bias.

  NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
  layer), and only scaled with a constant number, which is not trainable in
  this layer. However, the bias variable is trainable in this layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0),
              lr_multiplier = 1.0): # added
    super().__init__()
    fan_in = in_channels * kernel_size * kernel_size
    self.scale = gain / np.sqrt(fan_in)
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.lr_multiplier = lr_multiplier # added
  def forward(self, x):
    if x.ndim == 4:
      return x * self.scale + self.bias.view(1, -1, 1, 1)
    if x.ndim == 2:
      return x * self.scale + self.bias.view(1, -1)
    raise ValueError(f'The input tensor should be with shape [batch_size, '
                     f'channel, height, width], or [batch_size, channel]!\n'
                     f'But {x.shape} is received!')


class FirstBlock(nn.Module):
  """Implements the first block, which is a convolutional block."""

  def __init__(self,
               in_channels,
               out_channels,
               use_wscale=False,
               wscale_gain=np.sqrt(2.0),
               use_bn=False,
               activation_type='lrelu'):
    super().__init__()

    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
    self.scale = (wscale_gain / np.sqrt(in_channels * 3 * 3) if use_wscale else
                  1.0)
    self.bn = (BatchNormLayer(channels=out_channels) if use_bn else
               nn.Identity())

    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    return self.activate(self.bn(self.conv(x) * self.scale))


class ResBlock(nn.Module):
  """Implements the residual block.

  Usually, each residual block contains two convolutional layers, each of which
  is followed by batch normalization layer and activation layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               use_wscale=False,
               wscale_gain=np.sqrt(2.0),
               use_bn=False,
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
      kernel_size: Size of the convolutional kernels.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      use_wscale: Whether to use `wscale` layer.
      wscale_gain: The gain factor for `wscale` layer.
      use_bn: Whether to use batch normalization layer.
      activation_type: Type of activation. Support `linear` and `lrelu`.

    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
    super().__init__()

    # Add shortcut if needed.
    if in_channels != out_channels:
      self.add_shortcut = True
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)
      self.scale = wscale_gain / np.sqrt(in_channels) if use_wscale else 1.0
      self.bn = (BatchNormLayer(channels=out_channels) if use_bn else
                 nn.Identity())
    else:
      self.add_shortcut = False
      self.identity = nn.Identity()

    hidden_channels = min(in_channels, out_channels)

    # First convolutional block.
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=hidden_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    self.scale1 = (1.0 if use_wscale else
                   wscale_gain / np.sqrt(in_channels * 3 * 3))
    # NOTE: WScaleLayer is employed to add bias.
    self.wscale1 = WScaleLayer(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               gain=wscale_gain)
    self.bn1 = (BatchNormLayer(channels=hidden_channels) if use_bn else
                nn.Identity())

    # Second convolutional block.
    self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    self.scale2 = (1.0 if use_wscale else
                   wscale_gain / np.sqrt(hidden_channels * 3 * 3))
    self.wscale2 = WScaleLayer(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               gain=wscale_gain)
    self.bn2 = (BatchNormLayer(channels=out_channels) if use_bn else
                nn.Identity())

    if activation_type == 'linear':
      self.activate = nn.Identity()
    elif activation_type == 'lrelu':
      self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
      raise NotImplementedError(f'Not implemented activation function: '
                                f'{activation_type}!')

  def forward(self, x):
    if self.add_shortcut: # if in_channels != out_channels
      y = self.activate(self.bn(self.conv(x) * self.scale))
    else:
      y = self.identity(x)
    x = self.activate(self.bn1(self.wscale1(self.conv1(x) / self.scale1)))
    x = self.activate(self.bn2(self.wscale2(self.conv2(x) / self.scale2)))
    return x + y #??


class LastBlock(nn.Module):
  """Implements the last block, which is a dense block."""

  def __init__(self,
               in_channels,
               out_channels,
               use_wscale=False,
               wscale_gain=1.0,
               use_bn=False):
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)
    self.scale = wscale_gain / np.sqrt(in_channels) if use_wscale else 1.0
    self.bn = (BatchNormLayer(channels=out_channels) if use_bn else
               nn.Identity())

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x) * self.scale
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    return self.bn(x).view(x.shape[0], x.shape[1])


###-----------------------------------VGG16-----------------------
  
"""Contains the VGG16 model for perceptual feature extraction."""

# __all__ = ['VGG16', 'PerceptualModel']

# _WEIGHT_PATH = os.path.join(MODEL_DIR, 'vgg16.pth')
MODEL_DIR = r'D:\Hamed\InDomainGanInversion_D\models\pretrain'

_WEIGHT_PATH = os.path.join(MODEL_DIR, 'vgg16.pth')

_MEAN_STATS = (103.939, 116.779, 123.68) #h default mean for each channel?how to know?
# _MEAN_STATS = (103.939)

class VGG16(nn.Sequential):
  """Defines the VGG16 structure as the perceptual network.

  This models takes `RGB` images with pixel range [-1, 1] and data format `NCHW`
  as raw inputs. This following operations will be performed to preprocess the
  inputs (as defined in `keras.applications.imagenet_utils.preprocess_input`):
  (1) Shift pixel range to [0, 255].
  (3) Change channel order to `BGR`.
  (4) Subtract the statistical mean.

  NOTE: The three fully connected layers on top of the model are dropped.
  """

  def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Defines the network structure.

    Args:
      output_layer_idx: Index of layer whose output will be used as perceptual
        feature. (default: 23, which is the `block4_conv3` layer activated by
        `ReLU` function)
      min_val: Minimum value of the raw input. (default: -1.0)
      max_val: Maximum value of the raw input. (default: 1.0)
    """
    
    ## An OrderedDict is a dictionary subclass that remembers the order that keys were first inserted.
    # The only difference between dict() and OrderedDict() is that:
    # OrderedDict preserves the order in which the keys are inserted.
    # A regular dict doesn’t track the insertion order and iterating it gives the values in an arbitrary order.
    # By contrast, the order the items are inserted is remembered by OrderedDict.
    sequence = OrderedDict({
        'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        'layer1': nn.ReLU(inplace=True),
        'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        'layer3': nn.ReLU(inplace=True),
        'layer4': nn.MaxPool2d(kernel_size=2, stride=2),
        
        'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        'layer6': nn.ReLU(inplace=True),
        'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        'layer8': nn.ReLU(inplace=True),
        'layer9': nn.MaxPool2d(kernel_size=2, stride=2),
        
        'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        'layer11': nn.ReLU(inplace=True),
        'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer13': nn.ReLU(inplace=True),
        'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        'layer15': nn.ReLU(inplace=True),
        'layer16': nn.MaxPool2d(kernel_size=2, stride=2),
        
        'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        'layer18': nn.ReLU(inplace=True),
        'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer20': nn.ReLU(inplace=True),
        'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer22': nn.ReLU(inplace=True),
        'layer23': nn.MaxPool2d(kernel_size=2, stride=2),
        
        'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer25': nn.ReLU(inplace=True),
        'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer27': nn.ReLU(inplace=True),
        'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        'layer29': nn.ReLU(inplace=True),
        'layer30': nn.MaxPool2d(kernel_size=2, stride=2),
    })
    self.output_layer_idx = output_layer_idx
    self.min_val = min_val
    self.max_val = max_val
    self.mean = torch.from_numpy(np.array(_MEAN_STATS)).view(1, 3, 1, 1) #h float64, default for RGB
#     self.mean = torch.from_numpy(np.array(_MEAN_STATS)) #h float64
    self.mean = self.mean.type(torch.FloatTensor) #h float32
    super().__init__(sequence)

  def forward(self, x):
    x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val) #h from (-1,1) --> (0, 255)
    x = x.repeat(1, 3, 1, 1) # converting gray-scale image to RGB by repeating channels.
    x = x[:, [2, 1, 0], :, :] #h RGB --> BGR
#     x = x[:, 1, :, :] #h for gray scale
    x = x - self.mean.to(x.device)
    for i in range(self.output_layer_idx):
      x = self.__getattr__(f'layer{i}')(x)
    return x


# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 2

MAX_IMAGES_ON_RAM = 800

class PerceptualModel(object):
  """Defines the perceptual model class."""

  def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
    """Initializes."""
    self.use_cuda = USE_CUDA and torch.cuda.is_available()
    self.batch_size = MAX_IMAGES_ON_DEVICE
    self.ram_size = MAX_IMAGES_ON_RAM
    self.run_device = 'cuda' if self.use_cuda else 'cpu'
    self.cpu_device = 'cpu'

    self.output_layer_idx = output_layer_idx
    self.image_channels = 3 # default was 3 (RGB)
    self.min_val = min_val
    self.max_val = max_val
    self.net = VGG16(output_layer_idx=self.output_layer_idx,
                     min_val=self.min_val,
                     max_val=self.max_val)

    self.weight_path = _WEIGHT_PATH

    if not os.path.isfile(self.weight_path):
      raise IOError('No pre-trained weights found for perceptual model!')
    self.net.load_state_dict(torch.load(self.weight_path))
    self.net.eval().to(self.run_device)

  def get_batch_inputs(self, inputs, batch_size=None):
    """Gets inputs within mini-batch.

    This function yields at most `self.batch_size` inputs at a time.

    Args:
      inputs: Input data to form mini-batch.
      batch_size: Batch size. If not specified, `self.batch_size` will be used.
        (default: None)
    """
    total_num = inputs.shape[0]
    batch_size = batch_size or self.batch_size
    for i in range(0, total_num, batch_size):
      yield inputs[i:i + batch_size]

  def _extract(self, images):
    """Extracts perceptual feature within mini-batch."""
    if (images.ndim != 4 or images.shape[0] <= 0 or
        images.shape[0] > self.batch_size or images.shape[1] not in [1, 3]):
      raise ValueError(f'Input images should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`batch_size` no larger than {self.batch_size}, '
                       f'`channel` equals to 1 or 3!\n'
                       f'But {images.shape} is received!')
    if images.shape[1] == 1:
      images = np.tile(images, (1, 1, 1, 3))
    if images.shape[1] != self.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{images.shape[1]}, is not supported by the current '
                       f'perceptual model, which requires '
                       f'{self.image_channels} channels!')
    x = torch.from_numpy(images).type(torch.FloatTensor).to(self.run_device)
    f = self.net(x)
    return f.to(self.cpu_device).detach().numpy()

  def extract(self, images):
    """Extracts perceptual feature from input images."""
    if images.shape[0] > self.ram_size:
      self.logger.warning(f'Number of inputs on RAM is larger than '
                          f'{self.ram_size}. Please use '
                          f'`self.get_batch_inputs()` to split the inputs! '
                          f'Otherwise, it may encounter OOM problem!')

    results = []
    for batch_images in self.get_batch_inputs(images):
      results.append(self._extract(batch_images))

    return np.concatenate(results, axis=0)
