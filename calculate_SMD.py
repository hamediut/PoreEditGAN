"This script calculates the polytope functions for 2D or 3D images..."

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random
from src.SMDs import Microstructure, twoDCTimage2structure_mod, calculate_polytopes, calculate_smd_list

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
  
  parser.add_argument('--cpathPn', type= str, required = True)
  parser.add_argument('--runtimePn', type= str, required = True)
  parser.add_argument('--outputPn', type= str, required = True)

  parser.add_argument('--path_output', type =str, help= 'Path to the output folder to save dictinary containing the SMDs')
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
    
    img = tifffile.imread(os.path.join(args.path_input)).astype(np.uint8)
    print(f'Image shape: {img.shape}')
    poly_dict = {}
    list_polytopes = [ 'p3h', 'p3v','p4','p6', 'L']
    # list_polytopes = ['p3h','p4', 'L']
    base_poly_path = r'D:\Hamed\PSI\Results\compiled_poly_cpps'

    min_img_size = min(img.shape[1], img.shape[2]) if img.ndim==3 else min(img.shape[0], img.shape[1])

    # start_slice = 0

    # n_sample: image size
    par={'name':'polytopes','begx': 0, 'begy': 0, 'nsamp': min_img_size, 'edge_buffer': 0,
        'equalisation': False, 'equal_method': 'adaptive', 'stretch_percentile': 2,
        'clip_limit': 0.03, 'tvdnoise': False, 'tv_weight': 0.15, 'tv_eps': 2e-04,
        'median_filter': False, 'median_filter_length': 3,
        'thresholding_method': 'manual', 'thresholding_weight': 0.85, 'nbins': 256,
        'make_figs': False, 'fig_res': 400, 'fig_path':'./Plots/'}
    

    # cpathPn = r'D:\Hamed\PoreEditGAN_github\cpp_poly\512\Cpp_source\Polytope'
    # runtimePn = r'D:\Hamed\PoreEditGAN_github\cpp_poly\512\runtime/'
    # outputPn = r'D:\Hamed\PoreEditGAN_github\cpp_poly\512\runtime\output/'

    # cpathPn = os.path.join(args.path_cpp_code, str(min_img_size), 'Cpp_source\Polytope')
    # runtimePn = os.path.join(args.path_cpp_code, str(min_img_size), 'runtime/')
    # outputPn = os.path.join(args.path_cpp_code, str(min_img_size), 'runtime\output/')

    cpathPn = args.cpathPn
    runtimePn = args.runtimePn + '/'
    outputPn = args.outputPn + '/'



    print(f'cpathPn: {cpathPn}')
    print(f'runtimePn: {runtimePn}')
    print(f'outputPn: {outputPn}')


    for poly in list_polytopes:
        
        if poly == 'p2':
            
            s2, f2 = calculate_smd_list(img)
            poly_dict['s2'] = s2
            poly_dict['f2'] = f2
            print(f"S2 shape: {poly_dict['s2'][0].shape}")

        else:
            poly1, poly2 = calculate_polytopes(img, par, outputPn, cpathPn, runtimePn, polytope = poly)
            poly_dict[poly] = poly1
            # we determine the name of poly stored in the dict as key: f3h, f3v, f4, f6, fL
            poly2_name = poly.replace('p', 'f') if poly.startswith('p') else 'f' + poly
            poly_dict[poly2_name] = poly2
            print(f"{poly} shape: {poly_dict[poly][0].shape}")
        print(f'polytope {poly} done!')

    joblib.dump(poly_dict, os.path.join(args.path_output, 'SMDs.pkl'))
  

if __name__=="__main__":
    quantify_imgs()