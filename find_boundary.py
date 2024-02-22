"This script find a decision boundary between isolated and connected pores using Support vector machine"

import argparse
import pandas as pd
## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_log_csv', required= True, type=str,
                       help='Full path to the log file for inverting images. ')
  parser.add_argument('--dir_labels', type=str,
                        help='Full path to csv file containing labels')
  
#   parser.add_argument('--dir_img_3', type=str,
#                         help='Full path to 2D image taken on plane 3. it should be .tif file')
#   parser.add_argument('--RES', required= True, type = int,
#                       help = 'Representative image size' )
#   parser.add_argument('--train_img_size', type =int, default= 256,
#                       help = 'training image size, it can be smaller than RES.')
#   parser.add_argument('--img_channels', type= int, default= 1,
#                       help = 'number of channels. Default is 1 (binary image)')
#   parser.add_argument('--z_channels', type = int, default = 16,
#                       help = 'number of channles in noise vector.')
#   parser.add_argument('--z_size', type =int, default = 4,
#                       help ='size of noise vector. z dimension: (batch_size, z_channels, z_size, z_size, z_size)')
#   parser.add_argument('--num_train_imgs', type =int, default= 320000,
#                       help= 'Total number of training images from each 2D image for training')
#   parser.add_argument('--batch_size', type = int, default= 2,
#                       help= 'batch size for training the model. due to the gpu limitation we used 2. Use larger values if possible')
#   parser.add_argument('--D_batch_size', type = int, default=2, help= 'batch size for D')

  
  return parser.parse_args()


def get_boundary():
  args =parse_args()

  df_labels = pd.read_csv(args.dir_labels).iloc[:, :8]
  
  
  pass

if __name__=="__main__":
  get_boundary()