"""This script calculates two-point correlation function (S2) in 4D i.e., for more than one 3D images.
It get a path to folder where couple of 3D image with time exist.
Each 3D image in the folder shhould be a 3D tiff image file with shape of (num_slice, W, H).

Imges should be binary (0 and 1) with 1 for your feature of interest (e.g., pores or fractures).




Output:
This script saves two dictionaries, one for s2 and one for f2 (auto-scaled covariance) as 's2_3D_dic_r.pkl' and 'f2_3D_dic_r.pkl' by default.
Each dictionary has keys showing the experiment name and stack number that can be used to plot the results with time 
by getting the time corresponding to each stack number using timelog data. See jupyter notebook for more details.
"""

import os
import joblib
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import random
from src.SMDs import two_point_correlation3D, calculate_two_point_3D, cal_fn, omega_n, delta_omega
from src.SMDs import timelog_preprocessing

import tifffile

SEED = 43
random.seed(SEED)
np.random.seed(SEED)

## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_folder', required= True, type=str,
                       help='Path to the folder containing 3D images acquired in different time')
  parser.add_argument('--path_timelog', type = str, help = 'Full path to the timelog text file of the experiment.')
  parser.add_argument('--first_stack_number', type =int, default=67, help = 'The fist stack number to start from. Default value is for exp7 (Exp.1 in the paper)')
  parser.add_argument('--path_output', type =str, help= 'Path to the output folder to save dictinary containing radial S2')

  return parser.parse_args()

def S2_4d():
  args  = parse_args()

  s2_3D_dic_r = {}
  f2_3D_dic_r = {}

  # get the experiment name from the files in the folder.
  # for example in my case the image names are like:
  #Binary_KBr07_0001_f815_t1080_coordinates_541,575_size_512.tif
  # where KBr07 is the experiment name and '0001' is stack number 
  exp_name = os.listdir(args.path_folder)[0].split('_')[1]
  if exp_name == 'KBr07':
    first_scan = 67
  elif exp_name == 'KBr011':
    first_scan = 45
  else:
    first_scan = args.first_stack_number
    print(f'using default value for computation: {first_scan}')
  print(f'expeiment:{exp_name},  start computation from scan number {first_scan}')


#   print(exp_name)
  for idx, file in tqdm(enumerate(os.listdir(args.path_folder))):
       stack_num = file.split('_')[2] # getting the stack number
       if idx < first_scan:
         s2_3D_dic_r[f'{exp_name}_{stack_num}'] = 0
         f2_3D_dic_r[f'{exp_name}_{stack_num}'] = 0
        #  print(f'stack number: {stack_num}')
       else:
         img = tifffile.imread(os.path.join(args.path_folder, file)).astype(np.uint8)
         s2_3D_r = calculate_two_point_3D(img)
         s2_3D_dic_r[f'{exp_name}_{stack_num}'] = s2_3D_r
         f2_3D_dic_r[f'{exp_name}_{stack_num}'] = cal_fn(s2_3D_r, n=2)

  ## Saving the results of s2 and f2 in 4D separately
  joblib.dump(s2_3D_dic_r, os.path.join(args.path_output, 's2_3D_dic_r.pkl'))
  joblib.dump(f2_3D_dic_r, os.path.join(args.path_output, 'f2_3D_dic_r.pkl'))

#   s2_3D_dic_r = joblib.load(os.path.join(args.path_output, 's2_3D_dic_r.pkl'))
#   f2_3D_dic_r = joblib.load(os.path.join(args.path_output, 'f2_3D_dic_r.pkl'))

  #------------------------------- calculating omega
  ## we calculate omgea if there is more than one 3D image inside the path_folder
  if len(os.listdir(args.path_folder)) > 1:
    print('calculating omega metrics')
    omega_dict = {}
    
    ## for s2
    omega_dict['omega_s2_3d'] = omega_n(polytope= list(s2_3D_dic_r.values()))
    omega_dict['delta_omega_s2_3d'] = delta_omega(polytope= list(s2_3D_dic_r.values()))

    # for f2
    omega_dict['omega_f2_3d'] = omega_n(polytope= list(f2_3D_dic_r.values()))
    omega_dict['delta_omega_f2_3d'] = delta_omega(polytope= list(f2_3D_dic_r.values()))

    # save the dictionary containing the omega values for each 3D image in the input folder
    joblib.dump(omega_dict, os.path.join(args.path_output, 'omega_4D.pkl'))

    if args.path_timelog is not None:#if there is a timelog file i.e., you're running the code to reproduce our results
      timelog = timelog_preprocessing(args.path_timelog)

      for key in list(omega_dict.keys()):
        timelog[key] = omega_dict[key]
    
      file_suffix = 'Exp1' if exp_name == 'KBr07' else 'Exp2' if exp_name == 'KBr011' else 'Expn'
      timelog.to_csv(os.path.join(args.path_output, f'df_{file_suffix}.csv'))
            


    
    # omega_s2 = omega_n(polytope= list(s2_3D_dic_r.values()))
    # omega_del_s2 =  delta_omega(polytope= list(s2_3D_dic_r.values()))
    # omega_f2 = omega_n(polytope= list(f2_3D_dic_r.values()))
    # omega_del_f2 =  delta_omega(polytope= list(f2_3D_dic_r.values()))

  ## put the results of omega in a dataframe with time
  ## load and process the timelog as a dataframe
  
  
#   ## S2
#   timelog['omega_s2_3d'] =  omega_s2
#   timelog['delta_omega_s2_3d'] = omega_del_s2
  
#   ## F2
#   timelog['omega_f2_3d'] =  omega_f2
#   timelog['delta_omega_f2_3d'] = omega_del_f2


#   ## saving the dataframe as a csv file
 
#   file_suffix = 'Exp1' if exp_name == 'KBr07' else 'Exp2' if exp_name == 'KBr011' else 'Expn'
#   timelog.to_csv(os.path.join(args.path_output, f'df_{file_suffix}.csv'))


if __name__ == "__main__":
  S2_4d()
