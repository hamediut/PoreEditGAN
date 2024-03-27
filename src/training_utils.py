import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
from src.utils import _get_tensor_value

class MyClassDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform = None):
        self.labels_file = pd.read_csv(csv_file) # reading the csv file containing img_name and labels as in column HLP (human level performance)
        self.root_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return self.labels_file['HLP'].notna().sum() # this is total number of images for training
#         return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.labels_file.loc[index, 'img_name'] + '.tif')
        image = tifffile.imread(img_path).astype(np.uint8)
        image = np.stack((image,)*3, axis=-1)
        label = self.labels_file.loc[index, 'HLP']
        
        if self.transform:
            image = self.transform(image)
            
        return image , label
    
def classification_accuracy(pred, y):
    label_pred = pred.round()
    correct = label_pred.eq(y.view_as(label_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

##functions for inverting images-------------------------
def initial_code(img, E, G, device = 'cuda'):
    
    if type(img) ==  torch.Tensor:
        img = _get_tensor_value(img)
     
    # changing range of pixels values
    img = np.where(img < 1.0, -1.0, 1.0) # if pixels in {0,1} or {0, 255}
    
    
    if img.ndim == 2:#( res, res) like when you read images one by one --> (1, 1, res, res)
        img = img[np.newaxis][np.newaxis] 
        
    elif img.ndim == 3:#(batch, res, res) like when you read a couple of external images. --> (batch, 1, res, res)
        img = img[np.newaxis].transpose(0, 1)
    
    # np --> tensor: (b, c, res, res)
    img_tensor =  torch.from_numpy(img.astype(np.float32)).to(device)
    
    #E(x)
    encoded_img = E(img_tensor).view(img_tensor.shape[0], G.num_ws, G.z_dim) # (b, 16, 512)
#     encoded_img_np = _get_tensor_value(encoded_img).astype(np.float32)
    
    #G(E(X))
    
    recon_img = G.synthesis(encoded_img)
    
    
    return img_tensor, encoded_img, recon_img,


def load_cpk(checkpoint_fpath, module, device='cpu'):
    """
    
    checkpoint_path: path to load checkpoint from
    gen: an instance of generator that we want to load the state (what we've saved) into
    gen_opt: generator's optimizer we defined in previous training
    crit: an instance of critic that we want to load the state (what we've saved) into
    crit_opt: critic's optimizer we defined in previous training
    
    device: the device to load the trained model on. For inference, use cpu;otherwise you get an cuda out of memory error
    returns:
    gen, gen_opt, crit, crit_opt, step, s2_min
    """
    
    #load checkpoint
    checkpoint = torch.load(checkpoint_fpath, map_location= torch.device(device))
    
    #Generator----------------------------
    # initialize state_dict from checkpoint to generator:
    module.load_state_dict(checkpoint[f'enc_state_dict'])
    

    return module