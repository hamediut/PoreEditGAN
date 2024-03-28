import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from typing import List, Dict, Union
from tqdm import tqdm
from sklearn import svm
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error

from src.SMDs import calculate_smd_list,omega_n

import torch


def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()

def get_img_name_high(df_log:pd.DataFrame, mse_thresh: float = 1e-5 ) -> List[str]:


    
    """This function goes through the log file of inverting images and finds well-inverted images i.e.,
    whose mse error is lower than a threshold.
    """
    # Group by 'img_name' and find the minimum 'mse' for each group
    min_mse_by_img = df_log.groupby('img_name')['mse'].min().reset_index()
       # Filter rows where 'mse' is below the threshold
    well_inverted_imgs = min_mse_by_img[min_mse_by_img['mse'] < mse_thresh]
    # get the list of image names of high quality inversion
    list_imgs_name_high = well_inverted_imgs['img_name'].values
 
    return list_imgs_name_high


def get_codes_high(img_name_clean: List[str], latent_codes_dict: Dict[str, np.ndarray], res: int = 256) -> np.ndarray:

    """This function find the latent codes corresponding to the well-inverted images, high quality inversions
    
    parameters:
    img_name_clean: a dictionary containing images'name and the best mse (< thresh)
    latent_codes_dict: a dictionary containing images' name and their corresponding latent codes.

    returns:
    clean codes: a numpy of size (number of clean images, num_ws, w_dim)
    Note that num_ws depends on the image resolution:
    num_ws = 14 for res = 256,
    num_ws = 16 for res = 512,
    """
    ws_dim = (14 , 512) if res == 256 else (16, 512) if res ==512 else None

    latent_codes_clean= {key: value for key, value in latent_codes_dict.items() if key in img_name_clean}
    # numpy_arrays = [value for key, value in latent_codes_clean.items()]
    # latent_codes_np_high = np.squeeze(np.stack(numpy_arrays), axis = 1)

    ## fore res = 256, there are some clean images that are not in the latent_codes_dict, for mse 1e-5, 353 images
    # for this reason next line gives an error, but it is not necessary so commented out 
    
    # assert(latent_codes_np_high.shape[0] == len(img_name_clean))
    
    # print(f' {len(img_name_clean)} clean images are obtained.')
    # print(f' {latent_codes_np_high.shape[0]} latent codes are obtained.')

    return latent_codes_clean


def train_boundary(latent_codes, labels, split_ratio = 0.7):
    num_samples = latent_codes.shape[0]
    # latent_codes = latent_codes[:num_samples]
    labels = labels.reshape(-1,1)
    print(f'Latent codes shape: {latent_codes.shape}')
    print(f'Labels shape: {labels.shape}')
    
    
    # it seems that latent codes should be of dimension (num_samples, latent_space_dim)
    # so I concatenate all 16 channels of 512 for each image --> (num_samples, 16 * 512 = 8192)
    ## for res ==256 --> 14 * 512 = 7168
    # ws = 16 * 512 if res == 512 else 14 * 512 if res == 256 else None
    # latent_codes_concat =_get_tensor_value(torch.from_numpy(latent_codes).view(num_samples, ws))
    latent_codes_concat = latent_codes.reshape(num_samples, -1)
    # see how many 1 and 0 you have
    
    num_zeros = len(labels[labels == 0])
    num_ones = len(labels[labels == 1])
    print(f'Number of labels 0: {num_zeros}')
    print(f'Number of labels 1: {num_ones}')
    chosen_num = min(num_zeros, num_ones)
    # if there are 3 classes...
    if labels.max() > 1:
        num_twos = len(labels[labels == 2])
        print(f'Number of labels 2: {num_twos}')
        chosen_num = min(num_zeros, num_twos)
        
    
#     chosen_num = min(num_zeros, num_ones) # number of positive samples ( in our case label 1?)
    # from 500 samples, 259 of them are 1s and 241 of them are zero labels
    # therfore to have same numbe rof zeros and  ones we should choose a number =< 241
    split_ratio = split_ratio
    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    print(f'{chosen_num} samples are chosen from which number of training and validation samples are: {train_num}, {val_num}')

    
    sorted_idx = np.argsort(labels, axis=0)[::-1, 0] # sort the indices  from 1 to 0. 
    # The first indices (up to number of ones) correspond to label 1, the other are 0 
    latent_codes_sorted =  latent_codes_concat[sorted_idx] #sorting the codes the same way as labels
    labels_sorted =  labels[sorted_idx]
    
    # ones samples are the labels = 1 and their corresponding latent codes
    ones_idx = np.arange(chosen_num)
    np.random.shuffle(ones_idx) # random idx from 0 to chosen_num
    ones_train = latent_codes_sorted[:chosen_num][ones_idx[:train_num]] # (181, 8192) 70% of ones number = 0.7 * 259
    ones_val = latent_codes_sorted[:chosen_num][ones_idx[train_num:]] # (78, 8192) 30% of ones number = 0.3 * 259

    # zeros samples are the labels = 0 and their corresponding latent codes

    # Negative samples.
    zeros_idx = np.arange(chosen_num)
    np.random.shuffle(zeros_idx)
    zeros_train = latent_codes_sorted[-chosen_num:][zeros_idx[:train_num]]
    zeros_val = latent_codes_sorted[-chosen_num:][zeros_idx[train_num:]]

    print('------Size of training data---------')
    print(f'Ones shape: {ones_train.shape}')
    print(f'Zeros shape: {zeros_train.shape}')

    print('------Size of validation data---------')
    print(f'Ones shape: {ones_val.shape}')
    print(f'Zeros shape: {zeros_val.shape}')
    
    
    # Training set.
    train_data = np.concatenate([ones_train, zeros_train], axis = 0)
    train_label = np.concatenate([np.ones(train_num, dtype = int),
                                np.zeros(train_num, dtype = int)], axis = 0)
    print(f'Training: {train_num} ones, {train_num} zeros.')

    # Validation set.
    val_data = np.concatenate([ones_val, zeros_val], axis = 0)
    val_label = np.concatenate([np.ones(val_num, dtype = int),
                              np.zeros(val_num, dtype = int)], axis=0)
    print(f'Validation: {val_num} ones, {val_num} zeros.')
    print(f'In total-----------------------')
    print(f'Shape of training data: {train_data.shape}')
    print(f'Shape of val data: {val_data.shape}')
    print(f'Total number of samples used: {train_data.shape[0] + val_data.shape[0]}')
    
    # classification------------------------------
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    
    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        print(f'Accuracy for validation set: '
                    f'{correct_num} / {val_num * 2} = '
                    f'{correct_num / (val_num * 2):.6f}')
    
    a = classifier.coef_.reshape(1, latent_codes_concat.shape[1]).astype(np.float32)
    boundary = a/np.linalg.norm(a)
    print(f'Shape of boundary: {boundary.shape}')
    
    return classifier, boundary

###--------------------------------------------layer-wise manipulation-----------------------------
def get_layerwise_manipulation_strength(num_layers:int,
                                        truncation_psi: float,
                                        truncation_layers:int)->List[float]:
  """Gets layer-wise strength for manipulation.

  Recall the truncation trick played on layer [0, truncation_layers):

  w = truncation_psi * w + (1 - truncation_psi) * w_avg

  So, when using the same boundary to manipulate different layers, layer
  [0, truncation_layers) and layer [truncation_layers, num_layers) should use
  different strength to eliminate the effect from the truncation trick. More
  concretely, the strength for layer [0, truncation_layers) is set as
  `truncation_psi`, while that for other layers are set as 1.
  """
  strength = [1.0 for _ in range(num_layers)]
  if truncation_layers > 0:
    for layer_idx in range(0, truncation_layers):
      strength[layer_idx] = truncation_psi
  return strength


def parse_indices(obj, min_val=None, max_val=None)->List[int]:
  """Parses indices.

  If the input is a list or tuple, this function has no effect.

  The input can also be a string, which is either a comma separated list of
  numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
  be ignored.

  Args:
    obj: The input object to parse indices from.
    min_val: If not `None`, this function will check that all indices are equal
      to or larger than this value. (default: None)
    max_val: If not `None`, this function will check that all indices are equal
      to or smaller than this field. (default: None)

  Returns:
    A list of integers.

  Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
  """
  if obj is None or obj == '':
    indices = []
  elif isinstance(obj, int):
    indices = [obj]
  elif isinstance(obj, (list, tuple, np.ndarray)):
    indices = list(obj)
  elif isinstance(obj, str):
    indices = []
    splits = obj.replace(' ', '').split(',')
    for split in splits:
      numbers = list(map(int, split.split('-')))
      if len(numbers) == 1:
        indices.append(numbers[0])
      elif len(numbers) == 2:
        indices.extend(list(range(numbers[0], numbers[1] + 1)))
  else:
    raise ValueError(f'Invalid type of input: {type(obj)}!')

  assert isinstance(indices, list)
  indices = sorted(list(set(indices)))
  for idx in indices:
    assert isinstance(idx, int)
    if min_val is not None:
      assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
    if max_val is not None:
      assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

  return indices


def manipulate(latent_codes:np.ndarray,
               boundary:np.ndarray,
               start_distance:float =-5.0,
               end_distance:float =5.0,
               step: int =21,
               layerwise_manipulation:bool =False,
               num_layers:int =1,
               manipulate_layers:Union[bool, list, int, str, tuple] =None,
               is_code_layerwise:bool =False,
               is_boundary_layerwise: bool =False,
               layerwise_manipulation_strength:Union[int, float, tuple, list,] =1.0, device = 'cuda')->torch.Tensor:
  """Manipulates the given latent codes with respect to a particular boundary.

  Basically, this function takes a set of latent codes and a boundary as inputs,
  and outputs a collection of manipulated latent codes.

  For example, let `step` to be 10, `latent_codes` to be with shape [num,
  *code_shape], and `boundary` to be with shape [1, *code_shape] and unit norm.
  Then the output will be with shape [num, 10, *code_shape]. For each 10-element
  manipulated codes, the first code is `start_distance` away from the original
  code (i.e., the input) along the `boundary` direction, while the last code is
  `end_distance` away. Remaining codes are linearly interpolated. Here,
  `distance` is sign sensitive.

  NOTE: This function also supports layer-wise manipulation, in which case the
  generator should be able to take layer-wise latent codes as inputs. For
  example, if the generator has 18 convolutional layers in total, and each of
  which takes an independent latent code as input. It is possible, sometimes
  with even better performance, to only partially manipulate these latent codes
  corresponding to some certain layers yet keeping others untouched.

  NOTE: Boundary is assumed to be normalized to unit norm already.

  Args:
    latent_codes: The input latent codes for manipulation, with shape
      [num, *code_shape] or [num, num_layers, *code_shape].
    boundary: The semantic boundary as reference, with shape [1, *code_shape] or
      [1, num_layers, *code_shape].
    start_distance: Start point for manipulation. (default: -5.0)
    end_distance: End point for manipulation. (default: 5.0)
    step: Number of manipulation steps. (default: 21)
    layerwise_manipulation: Whether to perform layer-wise manipulation.
      (default: False)
    num_layers: Number of layers. Only active when `layerwise_manipulation` is
      set as `True`. Should be a positive integer. (default: 1)
    manipulate_layers: Indices of the layers to perform manipulation. `None`
      means to manipulate latent codes from all layers. (default: None)
    is_code_layerwise: Whether the input latent codes are layer-wise. If set as
      `False`, the function will first repeat the input codes for `num_layers`
      times before perform manipulation. (default: False)
    is_boundary_layerwise: Whether the input boundary is layer-wise. If set as
      `False`, the function will first repeat boundary for `num_layers` times
      before perform manipulation. (default: False)
    layerwise_manipulation_strength: Manipulation strength for each layer. Only
      active when `layerwise_manipulation` is set as `True`. This field can be
      used to resolve the strength discrepancy across layers when truncation
      trick is on. See function `get_layerwise_manipulation_strength()` for
      details. A tuple, list, or `numpy.ndarray` is expected. If set as a single
      number, this strength will be used for all layers. (default: 1.0)

  Returns:
    Manipulated codes, with shape [num, step, *code_shape] if
      `layerwise_manipulation` is set as `False`, or shape [num, step,
      num_layers, *code_shape] if `layerwise_manipulation` is set as `True`.

  Raises:
    ValueError: If the input latent codes, boundary, or strength are with
      invalid shape.
  """
  if not (boundary.ndim >= 2 and boundary.shape[0] == 1):
    raise ValueError(f'Boundary should be with shape [1, *code_shape] or '
                     f'[1, num_layers, *code_shape], but '
                     f'{boundary.shape} is received!')

  if not layerwise_manipulation:
    assert not is_code_layerwise
    assert not is_boundary_layerwise
    num_layers = 1
    manipulate_layers = None
    layerwise_manipulation_strength = 1.0

  # Preprocessing for layer-wise manipulation.
  # Parse indices of manipulation layers.
  layer_indices = parse_indices(
      manipulate_layers, min_val=0, max_val=num_layers - 1)
  if not layer_indices:
        layer_indices = list(range(num_layers))
  # Make latent codes layer-wise if needed.
  assert num_layers > 0
  if not is_code_layerwise:
    x = latent_codes[:, np.newaxis]
    x = np.tile(x, [num_layers if axis == 1 else 1 for axis in range(x.ndim)])
  else:
    x = latent_codes
    if x.shape[1] != num_layers:
      raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')
  # Make boundary layer-wise if needed.
  if not is_boundary_layerwise:
    b = boundary
    b = np.tile(b, [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
  else:
    b = boundary[0]
    if b.shape[0] != num_layers:
      raise ValueError(f'Boundary should be with shape [num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {b.shape} is received!')
  # Get layer-wise manipulation strength.
  if isinstance(layerwise_manipulation_strength, (int, float)):
    s = [float(layerwise_manipulation_strength) for _ in range(num_layers)]
  elif isinstance(layerwise_manipulation_strength, (list, tuple)):
    s = layerwise_manipulation_strength
    if len(s) != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{len(s)}` '
                       f'mismatches number of layers `{num_layers}`!')
  elif isinstance(layerwise_manipulation_strength, np.ndarray):
    s = layerwise_manipulation_strength
    if s.size != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{s.size}` '
                       f'mismatches number of layers `{num_layers}`!')
  else:
    raise ValueError(f'Unsupported type of `layerwise_manipulation_strength`!')
  s = np.array(s).reshape(
      [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
  b = b * s

  if x.shape[1:] != b.shape:
    raise ValueError(f'Latent code shape {x.shape} and boundary shape '
                     f'{b.shape} mismatch!')
  num = x.shape[0]
  code_shape = x.shape[2:]

  x = x[:, np.newaxis]
  b = b[np.newaxis, np.newaxis, :]
  l = np.linspace(start_distance, end_distance, step).reshape(
      [step if axis == 1 else 1 for axis in range(x.ndim)])
  results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])
  is_manipulatable = np.zeros(results.shape, dtype=bool)
  is_manipulatable[:, :, layer_indices] = True
  results = np.where(is_manipulatable, x + l * b, results)
  assert results.shape == (num, step, num_layers, *code_shape)
  interpolations = results if layerwise_manipulation else results[:, :, 0]
  # latent_codes.shape: (1, num_layers, z_dim), for res =256 -> (1, 14, 512)
  interpolations = torch.from_numpy(results).view(results.shape[1], latent_codes.shape[1], latent_codes.shape[2]).to(device)
  return interpolations


# device = 'cuda'
def best_start_distace_layerwise(G:torch.nn.Module,
                                 layers_strength: List[float],
                                 test_img:np.ndarray,
                                 test_code:np.ndarray,
                                 boundary:np.ndarray,
                                 start_search:float,
                                 end_search:float,
                                 num_steps:int = None,
                                 manip_lays:tuple = (10, 14),
                                 max_omega:float = 0.2,
                                 threshold_all:bool= False,
                                 plot:bool = True):
    
    """
    This function finds the best starting and ending points for manipulation by going generating
     images using the latent code for the image and manipulating it on semantic direction from 'start_search'
     to 'end_search' by taking 'num_steps' steps.

     The best start is the image reconstruction and is detemined by computing two-point correlation (s2) between
     real image and reconstructed image at different steps. The step that gives the minimum MSE between
     s2 of real and reconstructed image will be the best start.

     The best end determines how much we want to edit an image. for this we make use of Omega metric.
     This metric calculate the L1 distance between the starting s2 and the image reconstructed at each step.
     Then the best end is determined when we reach 'max_omega'. 
     To knwo how to set this 'max_omega', one should try manipulating a couple of images
     and see when manipulation goes off the latent space (see the paper for more details) 
    
    """
    
    if num_steps:
        steps = num_steps
    else:
        steps = end_search - start_search
    starting_values = np.linspace(start_search, end_search, steps)
    
    results = manipulate(test_code, boundary=boundary,
                         start_distance= start_search,
                         end_distance= end_search,
                         step= num_steps,
                         layerwise_manipulation= True, num_layers= 14,
                         manipulate_layers= list(range(manip_lays[0], manip_lays[1])),
                         is_code_layerwise= True, is_boundary_layerwise= True,
                         layerwise_manipulation_strength=layers_strength)
    
    interpolation_recons = _get_tensor_value(G.synthesis(results))
    
    
    ## threshold all
    if threshold_all:
        
        thresh = threshold_otsu(interpolation_recons)
        interpolation_recons_binary = np.where(interpolation_recons > thresh, 1, 0).astype(np.uint8)
    else:
        
        interpolation_recons_binary =  np.zeros(interpolation_recons[:,:,:,:].shape, dtype = np.uint8)

        for i in tqdm(range(interpolation_recons.shape[0])):

            thresh = threshold_otsu(interpolation_recons[i])
        #     thresh = 0
            interpolation_recons_thresh = np.where(interpolation_recons[i] > thresh, 1, 0)
            interpolation_recons_binary[i] = interpolation_recons_thresh[0].astype(np.uint8)

    if test_img.max()== 255:
        test_img_binary = np.where(test_img == 255, 1, 0).astype(np.uint8)
    else:
        test_img_binary = test_img
        
    s2_recon_list = []
    index_s2_mse = {}
    # index_px_mse = {}
    s2_test_img = calculate_smd_list(test_img_binary)[0]

    for i in tqdm(range(interpolation_recons_binary.shape[0])):


        s2_recon = calculate_smd_list(interpolation_recons_binary[i, 0, :, :])[0]
        s2_recon_list.append(s2_recon)
        mse_s2 = mean_squared_error(s2_test_img, s2_recon)
        # mse_px = mean_squared_error(test_img, interpolation_recons_binary[i, 0, :, :])
        
        index_s2_mse[i] = mse_s2
        # index_px_mse[i] = mse_px

    
    min_key_s2 = min(index_s2_mse, key=index_s2_mse.get)
    # min_key_px = min(index_px_mse, key=index_px_mse.get)
#     print("Key with minimum value of s2_mse=", min_key_s2)
#     print(f'Minimum s2_mse  = {index_s2_mse[min_key_s2]}')
    best_start = starting_values[min_key_s2]
    # best_start_px = starting_values[min_key_px]
    
    omega_s2_recon = omega_n(s2_recon_list)
#     return omega_s2_recon
    # for the last distance, we see when omega > 0.2, 0.2 is the max value observed during the experiment
    end_omega = max_omega if max(omega_s2_recon) > max_omega else max(omega_s2_recon)
#     return omega_s2_recon
    index_max_omega = np.where(np.array(omega_s2_recon)>= end_omega)[0][0]
    best_end = starting_values[index_max_omega]
    
    best_recon = interpolation_recons_binary[min_key_s2, 0, :, :]
    # best_recon_px = interpolation_recons_binary[min_key_px, 0, :, :]
#     print(f'Best start distance: {best_start}')
#     print(f'Best end distance: {best_end}')
    
    # if plot:
        
    #     fig, ax = plt.subplots(nrows=1, ncols= 3, sharex = True, sharey = True)
    #     ax[0].imshow(test_img_binary, cmap = 'gray')
    #     ax[1].imshow(best_recon, cmap = 'gray')
    #     ax[2].imshow(best_recon_px, cmap = 'gray')
    #     plt.show()
        
    #     plt.figure()
    #     s2_recon_s2 = calculate_smd_list(interpolation_recons_binary[min_key_s2 , 0, :, :])[0]
    #     plt.plot(s2_test_img, 'b', label = 'Real')
    #     plt.plot(s2_recon_s2, 'purple', label = 'Recon_s2')

    #     plt.legend()
    #     plt.show()
        
#     return best_start, best_end, s2_recon_list, omega_s2_recon
    return best_start, best_end, omega_s2_recon

