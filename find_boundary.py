"This script find a decision boundary between isolated and connected pores using Support vector machine"

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm

SEED = 43
random.seed(SEED)
np.random.seed(SEED)

from src.utils import get_img_name_high, get_codes_high, train_boundary

## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_log', required= True, type=str,
                       help='Full path to the log file for inverting images (the output of invert_imgs.py). ')
  parser.add_argument('--path_labels', required= True, type = str, help = 'Full path to the csv file containing image name and labels')
  parser.add_argument('--path_latents', required= True, type =str, help = 'Full path to the laten code file (.pkl)')
  
  parser.add_argument('--path_output', type =str, required=True, help= 'Where to save latent codes')


  parser.add_argument('--res', type=int, default= 512, help='size of images')
  
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


  ## reading the log file and clean it --> convert it into a dataframe file with column names etc

      ## --------------if such file doesn't exist, log.txt file should be cleaned...
  columns_list = ['img_name', 'step', 'loss_pix', 'loss_feat', 'loss_reg', 'loss', 'lr', 'mse']
  df_log = pd.read_csv(args.path_log, header = None) # up to image 5741
  df_log.columns = columns_list
    # #convert string values in each column to appropriate type...
    # # do this for each column. Note that each column has a different character to split('=', ':')

    # If you replace all ':' by '=', then you can run the following line to clean the df.
    # Note that there are a lot of lines that needs to be deleted before...(in visual studio)

  for column in columns_list:
        value_list = []
        for idx, value in tqdm(enumerate(list(df_log[column].values))):
    #         print(idx)
            if column == 'img_name':
                value_list.append(value.split('=')[1])
            elif column == 'step':
    #             print(idx)
                value_list.append(int(value.split('=')[1]))
            else:
                value_list.append(float(value.split('=')[1]))

        assert(len(value_list) == df_log.shape[0]) 
        df_log.loc[:, column] = value_list

  # df_log.to_csv(os.path.join(args.path_output, 'df_log_clean_test.csv'), index = False)
  

  # df_labels = pd.read_csv(os.path.join(args.dir_inputs, 'df_labels.csv'))[:-1] # Remove the last row because there is no latent code for that image
  df_labels = pd.read_csv(args.path_labels)[:-1] # labels obtained from the classifier
  print(df_labels.head())


  latent_codes_dict =  joblib.load(args.path_latents)
  # df_log = pd.read_csv(os.path.join(args.dir_inputs, 'df_log_clean_all.csv'))

  img_name_high = get_img_name_high(df_log, mse_thresh= args.mse_thresh)
  print(img_name_high)
  print(len(img_name_high))
  labels_high = [df_labels.loc[df_labels['img_name']== name, 'label'].values[0] for name in img_name_high]

  # labels_high = [df_labels.loc[df_labels['img_name']== name, 'preds_label'].values[0] for name in img_name_high] # with classifier
  # print(img_name_clean)
  latent_codes_high = get_codes_high(img_name_high, latent_codes_dict, res = args.res)

  classifier, boundary = train_boundary(latent_codes_high, np.array(labels_high), split_ratio= 0.7)

  joblib.dump(boundary, os.path.join(args.path_output, f'boundary_{args.mse_thresh}_class.pkl'))
  joblib.dump(classifier, os.path.join(args.path_output, f'classifier_{args.mse_thresh}_class.pkl'))
  joblib.dump(img_name_high, os.path.join(args.path_output, f'img_names_high_{args.mse_thresh}_class.pkl'))
 

if __name__=="__main__":
  get_boundary()