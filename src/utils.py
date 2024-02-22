import os
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from typing import List, Dict
from tqdm import tqdm
from sklearn import svm


def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()
def get_img_name_clean(df_log, mse_thresh = 1e-5 ):
    
    """This function goes through the log file of inverting images and finds well-inverted images i.e.,
    whose mse error is lower thatn a threshold.
    """
    
    list_img_name = list(np.unique(df_log['img_name'].values)) # get list of all images

    img_name_clean = {} # images with good recnonstruction
    for name in tqdm(list_img_name):
        min_s2 = df_log.loc[df_log['img_name'] == name]['mse'].min()

        if min_s2 < mse_thresh: #good results for <1e-6
            img_name_clean[name] = min_s2
 
    return img_name_clean


def get_clean_codes_labels(img_name_clean, latent_codes_dict, df_labels, res = 256):
    """This function find the labels and latent codes corresponding to the well-inverted images
    
    parameters:
    img_name_clean: a dictionary containing images'name and the best mse (< thresh)
    latent_codes_dict: a dictionary containing images' name and their corresponding latent codes.
    df_labels: the dataframe containing all images'name and their labels (with different labelling strategies)
    
    returns:
    clean codes: a numpy of size (number of clean images, num_ws, w_dim)
    clean labels: as a 1D numpy array
    Note that num_ws depends on the image resolution:
    num_ws = 14 for res = 256,
    num_ws = 16 for res = 512,
    """
    ws_dim = (14 , 512) if res == 256 else (16, 512) if res ==512 else None
    # get same keys in both dictionaries (with all images and clean images)
    similar_dict = set(list(img_name_clean.keys())).intersection(list(latent_codes_dict.keys()))
    # sort the codes as the above functions gets same dictionary with arbotrary order of key names
    list_codes_clean = sorted(list(similar_dict))
    
    latent_codes_np_clean = np.zeros((len(list_codes_clean), ws_dim[0], ws_dim[1]))
    labels_clean = []
    # labels_clean_idx = []
    for idx, name in enumerate(list_codes_clean):

        label = df_labels.loc[df_labels['img_name']== name, 'labels_3class'].values[0]
        labels_clean.append(label)
        latent_codes_np_clean[idx] = latent_codes_dict[name]
    
    assert(len(labels_clean) == latent_codes_np_clean.shape[0] == len(list_codes_clean))
    print(f'Reconstruction of {len(list_codes_clean)} was successful.')
    print(f'corresponding {len(labels_clean)} labels were selected.') 
    
    return latent_codes_np_clean, np.array(labels_clean), list_codes_clean

def train_boundary(latent_codes, labels, split_ratio = 0.7):
    num_samples = latent_codes.shape[0]
    # latent_codes = latent_codes[:num_samples]
    labels = labels.reshape(-1,1)
    print(f'Latent codes shape: {latent_codes.shape}')
    print(f'Labels shape: {labels.shape}')
    
    
    # it seems that latent codes should be of dimension (num_samples, latent_space_dim)
    # so I concatenate all 16 channels of 512 for each image --> (num_samples, 16 * 512 = 8192)
    ## for res ==256 --> 14 * 512 = 7168
    # ws = 16 * 512 if res == 512 else 14 * 512 if res == 256 else None
    # latent_codes_concat =_get_tensor_value(torch.from_numpy(latent_codes).view(num_samples, ws))
    latent_codes_concat = latent_codes.reshape(num_samples, -1)
    # see how many 1 and 0 you have
    
    num_zeros = len(labels[labels == 0])
    num_ones = len(labels[labels == 1])
    print(f'Number of labels 0: {num_zeros}')
    print(f'Number of labels 1: {num_ones}')
    chosen_num = min(num_zeros, num_ones)
    # if there are 3 classes...
    if labels.max() > 1:
        num_twos = len(labels[labels == 2])
        print(f'Number of labels 2: {num_twos}')
        chosen_num = min(num_zeros, num_twos)
        
    
#     chosen_num = min(num_zeros, num_ones) # number of positive samples ( in our case label 1?)
    # from 500 samples, 259 of them are 1s and 241 of them are zero labels
    # therfore to have same numbe rof zeros and  ones we should choose a number =< 241
    split_ratio = split_ratio
    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    print(f'{chosen_num} samples are chosen from which number of training and validation samples are: {train_num}, {val_num}')

    
    sorted_idx = np.argsort(labels, axis=0)[::-1, 0] # sort the indices  from 1 to 0. 
    # The first indices (up to number of ones) correspond to label 1, the other are 0 
    latent_codes_sorted =  latent_codes_concat[sorted_idx] #sorting the codes the same way as labels
    labels_sorted =  labels[sorted_idx]
    
    # ones samples are the labels = 1 and their corresponding latent codes
    ones_idx = np.arange(chosen_num)
    np.random.shuffle(ones_idx) # random idx from 0 to chosen_num
    ones_train = latent_codes_sorted[:chosen_num][ones_idx[:train_num]] # (181, 8192) 70% of ones number = 0.7 * 259
    ones_val = latent_codes_sorted[:chosen_num][ones_idx[train_num:]] # (78, 8192) 30% of ones number = 0.3 * 259

    # zeros samples are the labels = 0 and their corresponding latent codes

    # Negative samples.
    zeros_idx = np.arange(chosen_num)
    np.random.shuffle(zeros_idx)
    zeros_train = latent_codes_sorted[-chosen_num:][zeros_idx[:train_num]]
    zeros_val = latent_codes_sorted[-chosen_num:][zeros_idx[train_num:]]

    print('------Size of training data---------')
    print(f'Ones shape: {ones_train.shape}')
    print(f'Zeros shape: {zeros_train.shape}')

    print('------Size of validation data---------')
    print(f'Ones shape: {ones_val.shape}')
    print(f'Zeros shape: {zeros_val.shape}')
    
    
    # Training set.
    train_data = np.concatenate([ones_train, zeros_train], axis = 0)
    train_label = np.concatenate([np.ones(train_num, dtype = int),
                                np.zeros(train_num, dtype = int)], axis = 0)
    print(f'Training: {train_num} ones, {train_num} zeros.')

    # Validation set.
    val_data = np.concatenate([ones_val, zeros_val], axis = 0)
    val_label = np.concatenate([np.ones(val_num, dtype = int),
                              np.zeros(val_num, dtype = int)], axis=0)
    print(f'Validation: {val_num} ones, {val_num} zeros.')
    print(f'In total-----------------------')
    print(f'Shape of training data: {train_data.shape}')
    print(f'Shape of val data: {val_data.shape}')
    print(f'Total number of samples used: {train_data.shape[0] + val_data.shape[0]}')
    
    # classification------------------------------
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    
    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        print(f'Accuracy for validation set: '
                    f'{correct_num} / {val_num * 2} = '
                    f'{correct_num / (val_num * 2):.6f}')
    
    a = classifier.coef_.reshape(1, latent_codes_concat.shape[1]).astype(np.float32)
    boundary = a/np.linalg.norm(a)
    print(f'Shape of boundary: {boundary.shape}')
    
    return classifier, boundary