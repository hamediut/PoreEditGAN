""""This script invert images onto the latent space of a trained generator using a pretrained generator and encoder.
see the jupyter notebook 'InDStyleGAN2Inversion-MyImages_clean-new' in the stylegan3-main folder in David PC.
"""

import os
import joblib

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tifffile
import PIL
import torch
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error

import legacy
import dnnlib

from src.Networks import PerceptualModel, StyleGANEncoderNet

from dnnlib.util import Logger
from src.utils import _get_tensor_value
from src.training_utils import initial_code, load_cpk
from src.SMDs import calculate_smd_list

SEED = 43
random.seed(SEED)
np.random.seed(SEED)



## parsing arguments

def parse_args():
    """Pare arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_imgs', required = True, type = str, help= 'path to the folder containing images to invert')
    parser.add_argument('--path_G', required = True, help = 'Full path to the pre-trained Generator')
    parser.add_argument('--path_E', required = True, help = 'Full path to the pre-trained Encoder')
    parser.add_argument('--path_VGG', required= True, help = 'Full path to the pre-trained VGG model')
    parser.add_argument('--vgg_layer_idx', type=int, default=23, help='Index of VGG16 layer for perceptual loss')

    parser.add_argument('--res', type =int, default= 512, help= 'Image resolution. default = 512 by 512')

    parser.add_argument('--num_iter', type= int, default= 10000, help= 'Number of iteration to find the latent code')
    parser.add_argument('--lr', type = float, default = 0.05, help= 'learning rate for optimizer')
    parser.add_argument('--lr_decay_rate', type= float, default= 0.99, help= 'the degree of decaying learning rate')
    parser.add_argument('--lr_decay_step', type = int, default= 100, help = 'number of steps after which lr decays')

    ## Weights
    parser.add_argument('--recon_w', type = float, default= 4.0, help= 'The weight (coefficient) of pixel term in the loss function' )
    parser.add_argument('--feat_w', type = float, default= 5e-5, help= 'The weight of perceptual term in the loss i.e., distance in the feature space')
    parser.add_argument('--reg_w', type = float, default= 0.1, help= 'The weight of regularization term in the loss')
    
    ##thresholds to stop fine tuning the initial latent code from encoder
    parser.add_argument('--mse_thresh', type= float, default= 3e-6,
                         help= 'Threshold for mse between s2 correlation function of real image and recon image')
    parser.add_argument('--pixel_thresh', type = float, default = 0.1)

    parser.add_argument('--path_output', type =str, required=True, help= 'Where to save latent codes')
    return parser.parse_args()

def invert_imgs():

    args = parse_args()

    device = 'cuda'

    ## loading pre-trained networks: Generator, Encoder, and VGG

    ##Generator
    with dnnlib.util.open_url(args.path_G) as f:
        G_ema = legacy.load_network_pkl(f)['G_ema'].cuda().eval()

    print(f'Loading generator from: {args.path_G}')

    ## Encoder-------
    Enc = StyleGANEncoderNet(resolution= args.res).to(device)
    print(f'Loading encoder from: {args.path_E}')
    
    # loading the pretrained weights
    
    checkpoint= torch.load(args.path_E, map_location= torch.device(device))
    Enc.load_state_dict(checkpoint[f'enc_state_dict'])
    Enc.requires_grad_(False)

    ## VGG network: feature extractor

    # vgg_layer_idx = 23
    print(f'loading the vgg weight from: {args.path_VGG}')
    VGG = PerceptualModel(output_layer_idx= args.vgg_layer_idx, weight_path= args.path_VGG)
    # with 30 layers: (b, c, 512, 512) --> (b,c, 32, 32)
    ## for res =256:
        # idx = 30 --> 16
        # idx =23 --> 32
        # idx =16--> 64
    VGG.net.requires_grad_(False)
    
    
    assert next(G_ema.parameters()).requires_grad == False
    assert next(VGG.net.parameters()).requires_grad == False
    assert next(Enc.parameters()).requires_grad == False

    ##-------------------
    training_params = {
      'Images path': args.path_imgs, 'Output path': args.path_output,
      'Initial lr': args.lr, 'Decay rate':args.lr_decay_rate,'Decay step':args.lr_decay_step,
      'Recon weight':args.recon_w, 'perceptual weight':args.feat_w, 'Regularization weight': args.reg_w,
       'VGG layer index': args.vgg_layer_idx, 'MSE threshold': args.mse_thresh, 'pixel threshold': args.pixel_thresh,

      }
    
    Logger(file_name=os.path.join(args.path_output, 'log.txt'), file_mode='a', should_flush=True)
    print(f'Training parameters: {training_params}')
    
    
    # training loop

    codes = {}
    viz_results ={}

    pix_loss_list = []
    feat_loss_list = []
    lr_list = []

    mse_list =[]

    num_imgs = len(list(os.listdir(args.path_imgs)))

    for img_num, file in enumerate(list(os.listdir(args.path_imgs))):
                        
            if os.path.splitext(file)[1] == '.png':
                 img = np.array(PIL.Image.open(os.path.join(args.path_imgs, file)))
            elif os.path.splitext(file)[1] == '.tif':
                 img = tifffile.imread(os.path.join(args.path_imgs, file))
            # print(f'image shape = {img.shape}')
                 
                 
            mse = 1
            min_mse = 1
            
            # ## reading images...
            # if file.split('.')[1].lower() == 'png':
            #     img = np.array(PIL.Image.open(os.path.join(args.path_imgs, file)))
            # elif file.split('.')[1].lower() == 'tif':
            #     img = tifffile.imread(os.path.join(args.path_imgs, file))

            ##------------------------initial code---------
            real_tensor, init_code, init_recon = initial_code(img, Enc, G_ema)

            # here we use ecnoded images (E(x)) as a initial z instead of random z
            if img_num == 0:
                # for the first image we start with output of encoder
                z_init = init_code.detach()
                lr = args.lr
            else:
                # if the images are sequence, then for each slice we can start with the latent code of previous image
                last_code_saved = codes[list(codes.keys())[-1]]
        #         z_init = torch.from_numpy(best_code).to(device)
                z_init = torch.from_numpy(last_code_saved).to(device)
                lr = 0.01
                decayRate = 0.99 #0.98
            ##--------------------------
            z_init.requires_grad = True
            optimizer = torch.optim.Adam([z_init], lr = lr, betas=(0.5, 0.99) )
            z_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer= optimizer, gamma= args.lr_decay_rate)

            for step in range(args.num_iter):
                loss = 0

                #reconstruction loss (pixel loss)
                x_rec = G_ema.synthesis(z_init) # G(w)
                loss_pix = torch.mean( (real_tensor.to(device) - x_rec) ** 2)

                loss += loss_pix * args.recon_w

                # perceptual loss (feature loss)

                x_feat = VGG.net(real_tensor)
                x_rec_feat = VGG.net(x_rec)

                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                loss += loss_feat * args.feat_w

                ## regularization loss
                if args.reg_w:
                    z_enc = Enc(x_rec).view(1, G_ema.num_ws, G_ema.z_dim)
                    loss_reg = torch.mean( (z_init - z_enc) **2 )
                    loss += loss_reg * args.reg_w

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step > 0 and step % args.lr_decay_step == 0:
                    z_scheduler.step()
                
                ## calculate mse between s2 of real and recon imgs
                if step > 0 and step % 50 ==0:

                    real_np = np.where(_get_tensor_value(real_tensor)[0,0,:,:] > 0, 1, 0)
                    fake_np = _get_tensor_value(x_rec)[0,0,:,:]
                    fake_np = np.where(fake_np > threshold_otsu(fake_np), 1, 0)

                    real_s2 = calculate_smd_list(real_np)[0] #s2
                    fake_s2 = calculate_smd_list(fake_np)[0]
                    mse = mean_squared_error(real_s2, fake_s2)

                    if mse < min_mse:
                        best_code = _get_tensor_value(z_init)
                        codes[f'{os.path.splitext(file)[0]}'] = best_code
                        viz_results[f'{os.path.splitext(file)[0]}'] = fake_np

                        min_mse = mse
                    
                    print(f'Image_name={os.path.splitext(file)[0]}, Step={step}, Loss_pix = {_get_tensor_value(loss_pix):.3f}, loss_feat = {_get_tensor_value(loss_feat): .3f}, loss_reg = {_get_tensor_value(loss_reg):.3f}, loss = {_get_tensor_value(loss):.3f}, lr = {z_scheduler.get_last_lr()[-1]:0.4f}, mse = {mse:0.4e}' )
                    # 5e-6 was used for inverting exp 6 and exp7, but it should be less for sem images
                    if (mse < args.mse_thresh and loss_pix < args.pixel_thresh) or step == args.num_iter - 50: # 5e-7 for exp6&7 was good
                            fig, ax = plt.subplots(nrows= 1, ncols=2)
                            ax[0].imshow(img, cmap = 'gray')
                            ax[0].set_title('Real')
                            ax[1].imshow(viz_results[f'{os.path.splitext(file)[0]}'], cmap = 'gray')
                            ax[1].set_title('best recon')
                            fig.suptitle(f' Step = {step}, MSE = {mse:0.4e}, Pix_loss = {_get_tensor_value(loss_pix):.3f}')
                            plt.savefig(os.path.join(args.path_output, f'{os.path.splitext(file)[0]}.png'), dpi = 300)
                            # plt.show()

                            break
                
            if img_num > 0 and img_num % 50 ==0:
                 joblib.dump(codes, os.path.join(args.path_output, 'latent_codes.pkl'))
                #  joblib.dump(viz_results, os.path.join(args.path_output, 'viz_results_TraininImgs.pkl'))


if __name__=='__main__':
    invert_imgs()

