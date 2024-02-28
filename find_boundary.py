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
  parser.add_argument('--dir_inputs', required= True, type=str,
                       help='Full path to the log file for inverting images. ')
  parser.add_argument('--res', type=int, required= True, help='size of images')
  
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


def get_boundary():
  args =parse_args()

  df_labels = pd.read_csv(os.path.join(args.dir_inputs, 'df_labels.csv'))
  latent_codes_dict =  joblib.load(os.path.join(args.dir_inputs, 'latent_codes.pkl'))
  df_log = pd.read_csv(os.path.join(args.dir_inputs, 'df_log_clean_all.csv'))

  img_name_clean = get_img_name_clean(df_log, mse_thresh= args.mse_thresh)
  latent_codes_np_clean, labels_clean, list_codes_clean = get_clean_codes_labels(img_name_clean, latent_codes_dict, df_labels, res = args.res)

  classifier, boundary = train_boundary(latent_codes_np_clean, labels_clean, split_ratio= 0.7)

  joblib.dump(boundary, os.path.join(args.dir_inputs, f'boundary_{args.mse_thresh}.pkl'))
  joblib.dump(classifier, os.path.join(args.dir_inputs, f'classifier_{args.mse_thresh}.pkl'))
  joblib.dump(list_codes_clean, os.path.join(args.dir_inputs, f'clean_codes_{args.mse_thresh}.pkl'))

  print(f'Boundary shape: {boundary.shape}')
  # print(boundary)


if __name__=="__main__":
  get_boundary()