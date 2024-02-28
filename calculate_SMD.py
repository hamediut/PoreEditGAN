"This script calculates the polytope functions for 2D or 3D images..."

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random
from src.SMDs import Microstructure, twoDCTimage2structure_mod, calculate_polytopes

import tifffile

SEED = 43
random.seed(SEED)
np.random.seed(SEED)



## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_inputs', required= True, type=str,
                       help='Full path to the image. ')
  parser.add_argument('--path_cpp_code', type=str, required= True, help='Path to the folder containing compiled cpp codes for different image sizes')
  
#   parser.add_argument('--mse_thresh', type=float, default =1e-6,
#                         help='mse thrshold used to only get the latent codes of high quality inversions')
#   parser.add_argument('--RES', required= True, type = int,
#                       help = 'Representative image size' )
#   parser.add_argument('--train_img_size', type =int, default= 256,
#                       help = 'training image size, it can be smaller than RES.')
#   parser.add_argument('--img_channels', type= int, default= 1,
#                       help = 'number of channels. Default is 1 (binary image)')
#   parser.add_argument('--z_channels', type = int, default = 16,
#                       help = 'number of channles in noise vector.')

  
  return parser.parse_args()

def quantify_imgs():
    args = parse_args()
    
    img = tifffile.imread(os.path.join(args.dir_inputs)).astype(np.uint8)
    




    pass



if __name__=="__main__":
    quantify_imgs()