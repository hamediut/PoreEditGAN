"""This script calculates two-point correlation function (S2) in 4D i.e., for more than one 3D images.
It get a path to folder where couple of 3D image with time exist.
Each 3D image in the folder shhould be a 3D tiff image file with shape of (num_slice, W, H).

Imges should be binary (0 and 1) with 1 for your feature of interest (e.g., pores or fractures).




Output:
This script saves a dictionary ...
"""

import os
import joblib
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import random
from src.SMDs import two_point_correlation3D

import tifffile

SEED = 43
random.seed(SEED)
np.random.seed(SEED)

## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_input', required= True, type=str,
                       help='Full path to the image. ')
  parser.add_argument('--path_output', type =str, help= 'Path to the output folder to save dictinary containing the S2 in different directions')

  return parser.parse_args()

def S2_3d():
  args  = parse_args()

  img = tifffile.imread(os.path.join(args.path_input)).astype(np.uint8)
  print(f'Image shape: {img.shape}')

  if img.ndim != 3:
    raise ValueError(
      f"Image shape = {img.shape}"
      f"but input image should be 3D"
    )
  
  Nr = min(img.shape)//2 # min of shape, in case of non-cubic images (x=266, y = 512, z= 512)
  two_point_covariance = {}
  for j, direc in tqdm(enumerate( ["x", "y", "z"]) ):
                     
            two_point_direc =  two_point_correlation3D(img, dim = j, var = 1)
            two_point_covariance[direc] = two_point_direc
#         Nr = two_point_covariance[direc].shape[0]// 2

  direc_covariances = {}
  for direc in ["x", "y", "z"]:
            direc_covariances[direc] =  np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[:Nr]
        
#   average_yz = ( np.array(direc_covariances['y']) + np.array(direc_covariances['z']) )/2
  ## calculate radial S2 by averaging along x, y, z
  s2_r = (np.array(direc_covariances['x']) + np.array(direc_covariances['y']) + np.array(direc_covariances['z']) )/3
  ## put them all in a dictionary
  s2_dict = {}
  s2_dict['x'] = np.array(direc_covariances['x'])
  s2_dict['y'] = np.array(direc_covariances['y'])
  s2_dict['z'] = np.array(direc_covariances['z'])
  s2_dict['r'] = s2_r

  # save the dictionary in the output path
  joblib.dump(s2_dict, os.path.join(args.path_output, 's2_3D.pkl'))
#   return np.array(direc_covariances['x']), np.array(direc_covariances['y']), np.array(direc_covariances['z'])

if __name__ == "__main__":
  S2_3d()
