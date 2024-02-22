"This script find a decision boundary between isolated and connected pores using Support vector machine"

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random

SEED = 43
random.seed(SEED)
np.random.seed(SEED)

from src.utils import get_img_name_clean, get_clean_codes_labels, train_boundary

## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_inputs', required= False, type=str,
                       help='Full path to the log file for inverting images. ')
  parser.add_argument('--res', type=int,
                        help='size of images')
  
  parser.add_argument('--mse_thresh', type=float, default =1e-6,
                        help='mse thrshold used to only get the latent codes of high quality inversions')
#   parser.add_argument('--RES', required= True, type = int,
#                       help = 'Representative image size' )
#   parser.add_argument('--train_img_size', type =int, default= 256,
#                       help = 'training image size, it can be smaller than RES.')
#   parser.add_argument('--img_channels', type= int, default= 1,
#                       help = 'number of channels. Default is 1 (binary image)')
#   parser.add_argument('--z_channels', type = int, default = 16,
#                       help = 'number of channles in noise vector.')

  
  return parser.parse_args()