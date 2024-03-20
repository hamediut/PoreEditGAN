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

from src.utils import get_img_name_high, get_codes_high, train_boundary

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

  # df_labels = pd.read_csv(os.path.join(args.dir_inputs, 'df_labels.csv'))[:-1] # Remove the last row because there is no latent code for that image
  df_labels = pd.read_csv(os.path.join(args.dir_inputs, 'labels.csv'))[:-1] # labels obtained from the classifier
  latent_codes_dict =  joblib.load(os.path.join(args.dir_inputs, 'latent_codes.pkl'))
  df_log = pd.read_csv(os.path.join(args.dir_inputs, 'df_log_clean_all.csv'))

  img_name_high = get_img_name_high(df_log, mse_thresh= args.mse_thresh)
  # labels_high = [df_labels.loc[df_labels['img_name']== name, 'labels_3class'].values[0] for name in img_name_high]
  labels_high = [df_labels.loc[df_labels['img_name']== name, 'preds_label'].values[0] for name in img_name_high] # with classifier
  # print(img_name_clean)
  latent_codes_high = get_codes_high(img_name_high, latent_codes_dict, res = args.res)

  classifier, boundary = train_boundary(latent_codes_high, np.array(labels_high), split_ratio= 0.7)

  joblib.dump(boundary, os.path.join(args.dir_inputs, f'boundary_{args.mse_thresh}_class.pkl'))
  joblib.dump(classifier, os.path.join(args.dir_inputs, f'classifier_{args.mse_thresh}_class.pkl'))
  joblib.dump(img_name_high, os.path.join(args.dir_inputs, f'img_names_high_{args.mse_thresh}_class.pkl'))
 

if __name__=="__main__":
  get_boundary()