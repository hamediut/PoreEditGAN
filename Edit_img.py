"This script find a decision boundary between isolated and connected pores using Support vector machine"

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import random
import tifffile
import PIL
from PIL import Image
import torch
from tqdm import tqdm
from skimage.filters import threshold_otsu


SEED = 43
random.seed(SEED)
np.random.seed(SEED)


import legacy
import dnnlib
from src.utils import _get_tensor_value, get_layerwise_manipulation_strength, parse_indices, manipulate, best_start_distace_layerwise
# from src.SMDs import get_layer
# from src import torch_utils


## parsing arguments
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir_imgs', required= True, type=str,
                       help='Path to the folder containing images you want to edit. ')
  
  parser.add_argument('--dir_latents', required= True, type=str,
                       help='Path to latent codes pkl file. ')
#   parser.add_argument('--res', type=int,
#                         help='size of images')
  
  parser.add_argument('--dir_boundary', required = True, type= str, help = 'Full path to the boundary file *.pkl')
  parser.add_argument('--dir_classifier', required = False, type= str, help = 'Full path to the classifier file: *.pkl')
  
#   parser.add_argument('--max_omega', type=float, default =0.15, help = 'Maximum value for omega to determine the maximum editing')
  parser.add_argument('--G_pkl', required = True, help = 'Full path to the pre-trained')
  parser.add_argument('--dir_output', required = True, help = 'Full path to the pre-trained')
  parser.add_argument('--layerwise_manip', action= 'store_true', help= 'Whether to do layer-wise manipulation or not.\
                       if the argument is passed in command line, it will set to True, otherwise False')
  parser.add_argument('--truncation_psi', type = float, help='Truncation value used for truncating the latent code')
  parser.add_argument('--truncation_layers', type= int, help='The number of layers you want to apply truncation.it starts from first layer up to the specified layer here.\
                       Other layers will have a truncation of 1 which means no truncation.')
  parser.add_argument('--manip_layers', nargs= 2, type = int, help = 'range of layers to manipulate, received as 2 values \
                       e.g., 10 14 means only layers 10 to 14 are used to manipulate an image')
  parser.add_argument('--max_omega', type = float, default = 0.2, help= 'Maximum omega to determine wjere to stop editing the image.')

#   parser.add_argument('--RES', required= True, type = int,
#                       help = 'Representative image size' )
#   parser.add_argument('--train_img_size', type =int, default= 256,
#                       help = 'training image size, it can be smaller than RES.')
#   parser.add_argument('--img_channels', type= int, default= 1,
#                       help = 'number of channels. Default is 1 (binary image)')
#   parser.add_argument('--z_channels', type = int, default = 16,
#                       help = 'number of channles in noise vector.')

  
  return parser.parse_args()

def connect_pores():
  args =parse_args()

  # Load dictionary of latent codes: {'image_name': latent_code}
  latent_codes = joblib.load(args.dir_latents)
  # load the classifier
  classifier = joblib.load(args.dir_classifier)
  boundary = joblib.load(args.dir_boundary)
  print(f'Boundary shape : {boundary.shape}')
  ##load the pre-trained generator
#   G_ema = joblib.load(args.G_pkl)
  with dnnlib.util.open_url(args.G_pkl) as f:
        G_ema = legacy.load_network_pkl(f)['G_ema'].cuda().eval()
  
  # reading images
  list_imgs = [file for file in os.listdir(args.dir_imgs) if os.path.splitext(file)[1] in ['.tif', '.png'] ]

  for img_name in list_imgs:
    img = tifffile.imread(os.path.join(args.dir_imgs, img_name)) if img_name.endswith('.tif') else np.array(PIL.Image.open(os.path.join(args.dir_imgs, img_name)))
    res = img.shape[0] # image resolution
    code = latent_codes[os.path.splitext(img_name)[0]].reshape(1, -1) # reshape it from (16, 512) to (1, 16*512)
    label_pred = classifier.predict(code)

    print(f'Predicted label for image {os.path.splitext(img_name)[0]}: {label_pred}')

    if args.layerwise_manip or res ==256:
       layers_strength =  get_layerwise_manipulation_strength(num_layers= G_ema.num_ws, truncation_psi= args.truncation_psi, truncation_layers= args.truncation_layers)
       manip_lays = tuple(args.manip_layers) #(10, 14)
       boundary = _get_tensor_value(torch.from_numpy(boundary).view(1, G_ema.num_ws, G_ema.z_dim))
       code = latent_codes[os.path.splitext(img_name)[0]] # without reshaping--> (1, 14, 512)
       print(f'Boundary shape : {boundary.shape}')
       best_start, best_end, _ = best_start_distace_layerwise(G_ema, layers_strength, img, code, boundary,
                                                              start_search=-50.0, end_search= 1500.0,
                                                              num_steps = 300, manip_lays= manip_lays, max_omega=args.max_omega)
       
       ## manipulate the latent code
       manipulated_code = manipulate(latent_codes=code,
                                     boundary=boundary,
                                     start_distance= best_start, end_distance= best_end,step=300,
                                     layerwise_manipulation= True, num_layers = G_ema.num_ws,
                                     manipulate_layers=list(range(manip_lays[0], manip_lays[1])),
                                     is_code_layerwise= True, is_boundary_layerwise= True,
                                     layerwise_manipulation_strength=layers_strength)
       
       interpolation_recons = _get_tensor_value(G_ema.synthesis(manipulated_code))
    else:
       ## steps in latent space
       start_distance = 0
       end_distance = 20
       steps = 10
       linspace1 = np.linspace(start_distance, end_distance, steps)
       linspace1 = linspace1 - code.dot(boundary.T)
       
       start_distance = 21
       end_distance = 70 #50
       steps = 70
       linspace2 = np.linspace(start_distance, end_distance, steps)
       linspace2 = linspace2 - code.dot(boundary.T)
       
       linspace = np.concatenate((linspace1, linspace2), axis = 1)
       linspace = linspace.reshape(-1, 1).astype(np.float32)
       ## editing...G(z +n)
       interpolation = code + linspace * boundary.reshape(1, 1, -1)
       device = 'cuda'
       interpolation = torch.from_numpy(interpolation).view(interpolation.shape[1], G_ema.num_ws, G_ema.w_dim).to(device)
       interpolation_recons = _get_tensor_value(G_ema.synthesis(interpolation))

    ## thresholding the edited images
    interpolation_recons_binary =  np.zeros(interpolation_recons[:,0,:,:].shape, dtype = np.uint8)

    for i in tqdm(range(interpolation_recons.shape[0])):
        
        thresh = threshold_otsu(interpolation_recons[i])
    #     thresh = 0.5
        interpolation_recons_thresh = np.where(interpolation_recons[i] > thresh, 1, 0)
        interpolation_recons_binary[i] = interpolation_recons_thresh[0].astype(np.uint8)
    
    tifffile.imsave(os.path.join(args.dir_output ,f'{os.path.splitext(img_name)[0]}_boundary_1e5.tif'), interpolation_recons_binary)


if __name__=='__main__':
  connect_pores()