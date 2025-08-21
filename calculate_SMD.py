"""This script calculates the polytope functions for 2D binary images.
The input image can be one single 2D image file, or a 3D tiff image file with shape of (num_slice, W, H).

Imges should be binary (0 and 1) with 1 for your feature of interest (e.g., pores or fractures).

The cpp code located in "Cpp_source-> Polytope->Sample_Pn_UU.cpp" is compiled for images with W=H=512,
if you have images of different size, you should do the following changes in cpp code (Sample_Pn_UU.cpp) and recompile it:

#define MAXX 513 --> change this to your image_size + 1 (e.g., for image size of 256, it should be 257)
#define Nt 256 --> change this to half of you image size (e.g., for image size of 256, it should be 128)


Output:
This script saves a dictionary for each image in the output folder specified by user.
In each dictionary, the polytope functions (s2, p3, p4, ..L, f2, f3, f4, fL) are the keys and values are the probabilities at each distance r.
See the jupyter notebook "polytopes_example.ipynb" for examples of plotting SMDs after running this script.
"""

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random
from src.SMDs import Microstructure, twoDCTimage2structure_mod, calculate_polytopes, calculate_smd_list, calculate_cluster_fn
from src.SMDs import omega_n, timelog_preprocessing

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
  # parser.add_argument('--path_cpp_code', type=str, required= True, help='Path to the folder containing compiled cpp codes for different image sizes')
  parser.add_argument('--image_size', type = int, default= 512, help = 'The size of image slices')
  parser.add_argument('--cpathPn', type= str, required = True, help = 'path to polytope folder in cpp_source')
  parser.add_argument('--runtimePn', type= str, required = True, help = 'path to runtime folder in cpp_source')
  parser.add_argument('--outputPn', type= str, required = True, help = 'path to output folder in runtime')
  parser.add_argument('--path_timelog', required= True, type = str, help = 'Full path to the timelog text file of the experiment.')


  parser.add_argument('--path_output', type =str, help= 'Path to the output folder to save dictinary containing the SMDs')


  
  return parser.parse_args()

def quantify_imgs():
    args = parse_args()

    img = tifffile.imread(os.path.join(args.path_input)).astype(np.uint8)
    # img = img[::5, :, :] # just to test the code
    print(f'Image shape: {img.shape}')
    min_img_size = min(img.shape[1], img.shape[2]) if img.ndim ==3 else min(img.shape)

    if min_img_size != 512:
        raise ValueError(
            f"Detected image min dimension = {min_img_size}, but the cpp code "
            f"is compiled for 512 by default.\n"
            f"Please recompile the cpp code with:\n"
            f"    MAXX = {min_img_size + 1}\n"
            f"    Nt   = {min_img_size // 2}\n"
            f"\nSee the docstring and branch documentation for instructions."
        )

    # path to cpp folders, this works when the full path is provided, not reelative path
    cpathPn = args.cpathPn
    runtimePn = args.runtimePn + '/'
    outputPn = args.outputPn + '/'

    print(f'cpathPn: {cpathPn}')
    print(f'runtimePn: {runtimePn}')
    print(f'outputPn: {outputPn}')


    par={'name':'polytopes','begx': 0, 'begy': 0, 'nsamp': min_img_size, 'edge_buffer': 0,
    'equalisation': False, 'equal_method': 'adaptive', 'stretch_percentile': 2,
    'clip_limit': 0.03, 'tvdnoise': False, 'tv_weight': 0.15, 'tv_eps': 2e-04,
    'median_filter': False, 'median_filter_length': 3,
    'thresholding_method': 'manual', 'thresholding_weight': 0.85, 'nbins': 256,
    'make_figs': False, 'fig_res': 400, 'fig_path':'./Plots/'}

    # list_polytopes = ['p2', 'p3h', 'p3v','p4','p6', 'L', 'c2'] # list of polytopes to compute
    list_polytopes = ['p2'] # list of polytopes to compute

    poly_dict = {}
    for poly in list_polytopes:
        
        if poly == 'p2':
            
            s2, f2 = calculate_smd_list(img)
            poly_dict['s2'] = s2
            poly_dict['f2'] = f2
            print(f"S2 shape: {poly_dict['s2'][0].shape}")
        elif poly == 'c2':
            c2 = calculate_cluster_fn(img, start_scan= 0)
            poly_dict['c2'] = c2
        else:
            poly1, poly2 = calculate_polytopes(img, par, outputPn, cpathPn, runtimePn, polytope = poly)
            poly_dict[poly] = poly1
            # we determine the name of poly stored in the dict as key: f3h, f3v, f4, f6, fL
            poly2_name = poly.replace('p', 'f') if poly.startswith('p') else 'f' + poly
            poly_dict[poly2_name] = poly2
            print(f"{poly} shape: {poly_dict[poly][0].shape}")
        print(f'polytope {poly} done!')

    # joblib.dump(poly_dict, os.path.join(args.path_output, 'SMDs.pkl'))

    # poly_dict = joblib.load(os.path.join(args.path_output, 'SMDs.pkl'))

    # ## ------------------calculating Omega-----
    omega_dict = {}
    for key in list(poly_dict.keys()):
        omega_dict[key] = omega_n(poly_dict[key])
    # omega_dict['s2'] = omega_n(poly_dict['s2'])
    # omega_dict['p3'] = omega_n(poly_dict['p3h'])
    # omega_dict['p4'] = omega_n(poly_dict['p4'])
    # omega_dict['p6'] = omega_n(poly_dict['p6'])
    # omega_dict['L'] = omega_n(poly_dict['L'])
    # omega_dict['c2'] = omega_n(poly_dict['c2'])

    # joblib.dump(omega_dict, os.path.join(args.path_output, 'omega_SMDs.pkl'))

# ## calculating omega and put them in the dataframe
    # omega_s2 = omega_n(poly_dict['s2'])
#     omega_p3 = omega_n(poly_dict['p3h'])
#     omega_p4 = omega_n(poly_dict['p4'])
#     omega_p6 = omega_n(poly_dict['p6'])
#     omega_L = omega_n(poly_dict['L'])
#     omega_c2 = omega_n(poly_dict['c2'])
 
    ## load and process the timelog as a dataframe
    # here we check if the number of images match the numbe rof rows in the timelog,
    # if so, we put the results of omega in a dataframe with time.
    # Otherwise, we save the results as a dictionary within a pickle file
    timelog = timelog_preprocessing(args.path_timelog)
    if timelog.shape[0] == img.shape[0]:
        print(f'Number of images match the timelog...saving the results as a .csv file in: {args.path_output}')
        for key in list(omega_dict.keys()):
            timelog[key] = omega_dict[key]
        # timelog['p3'] = omega_dict['p3']
        # timelog['p4'] = omega_dict['p4']
        # timelog['p6'] = omega_dict['p6']
        # timelog['L'] = omega_dict['L']
        # timelog['c2'] = omega_dict['c2']
        ##saving the datafram
        timelog.to_csv(os.path.join(args.path_output, f'df_SMDs_2D_omega.csv'))
    else:
        print(f'Number of images do not match the timelog...saving the results as a .pkl file in: {args.path_output}')

        joblib.dump(omega_dict, os.path.join(args.path_output, 'omega_SMDs.pkl'))




if __name__=="__main__":
    quantify_imgs()