import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tifffile

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