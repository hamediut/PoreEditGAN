import os
from glob import glob
import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
import copy
import tifffile

import joblib

from torch.utils.data import Dataset, DataLoader
from skimage import io
import time
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm
# from tqdm.notebook import tqdm, trange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import random

from src.utils import _get_tensor_value
from src.training_utils import MyClassDataset, classification_accuracy


### setting the seeds------------
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
    # Configure PyTorch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
###-----------------------------

def accuracy_classifier(pred, y):
    label_pred = pred.round()
    correct = label_pred.eq(y.view_as(label_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_imgs', required= True, type=str,
                       help='Path to the folder containing images you want to edit. ')
  
  parser.add_argument('--path_labels', required= True, type=str,
                       help='Path to the csv files containing image names and manual labels ')
  
  parser.add_argument('--lr', type= float, default = 0.001, help= 'Learning rate for adam optimizer')
  parser.add_argument('--path_outputs', type =str, required= True, help = 'Path to the folder to save outputs')

  parser.add_argument('--num_epochs', type = int, default= 50, help= 'Number of epochs to fme-tune the resnet50 model.')
#   parser.add_argument('--res', type=int,
#                         help='size of images')
  
#   parser.add_argument('--dir_boundary', required = True, type= str, help = 'Full path to the boundary file *.pkl')
#   parser.add_argument('--dir_classifier', required = False, type= str, help = 'Full path to the classifier file: *.pkl')
  
# #   parser.add_argument('--max_omega', type=float, default =0.15, help = 'Maximum value for omega to determine the maximum editing')
#   parser.add_argument('--G_pkl', required = True, help = 'Full path to the pre-trained')
#   parser.add_argument('--dir_output', required = True, help = 'Full path to the pre-trained')

  return parser.parse_args()

def fine_tune_resnet50()-> None:
   args = parse_args()

   ## define image transformation
   img_transform = transforms.Compose([           
                transforms.ToPILImage(),  
                transforms.Resize(256),                   
                transforms.CenterCrop(224),               
                transforms.ToTensor(),                  
 ])
   

   # constructing the dataset
   mydataset = MyClassDataset(img_dir = args.path_imgs,
                              csv_file= args.path_labels,
                              transform = img_transform)
   
   
   num_train = int(mydataset.__len__() * 0.85)
   num_test = mydataset.__len__() - num_train
   ## We have 600 labeled images (HLP column in the csv file)
   train_set, valid_set = torch.utils.data.random_split(mydataset, [num_train, num_test])
   train_loader = DataLoader(dataset = train_set, batch_size = 16, shuffle = True)
   valid_loader = DataLoader(dataset = valid_set, batch_size = 16, shuffle = False)

   # ##test the dataloader------
   # test_img, test_label = mydataset[503]

   # print(test_img.shape,test_label)

   # plt.figure()
   # plt.imshow(_get_tensor_value(test_img)[0, :,:], cmap = 'gray')
   # plt.savefig(os.path.join(args.path_outputs, 'test_img.png'), dpi = 300)

   ##-----------------------------------
   #Load pretrained ResNet50 Model
   resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
   # Freeze model parameters
   for param in resnet50.parameters():
      param.requires_grad = False

   # Change the final layer of ResNet50 Model for Transfer Learning
   fc_inputs = resnet50.fc.in_features # 2048
   resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    #nn.Dropout(0.4),
    nn.Linear(128, 1), # Since 2 possible outputs
    nn.Sigmoid()
)
   ###--------------------------------------
   ## define the model, loss function, optimizer
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = resnet50.to(device)
   

   # def initialize_weights(m):
   #  if isinstance(m, nn.Linear):
   #      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   #      if m.bias is not None:
   #          nn.init.constant_(m.bias, 0)

   # Define Optimizer and Loss Function
   criterion = nn.BCELoss()
   optimizer = optim.Adam(model.parameters(), lr= args.lr)
   
   # model = model.to(device)
   criterion = criterion.to(device)

   ###-------------------------------------
   EPOCHS = args.num_epochs
   max_acc = 0.0
   best_model_wts = copy.deepcopy(model.state_dict())

   history = {
      'train_loss': [],
      'train_acc': [], 
      'val_loss': [],
      'val_acc': []
   }

   for epoch in range(EPOCHS):

      ##train
      model.train()
      train_loss_list = []
      train_acc_list = []
      for train_img, train_label in train_loader:
          
          optimizer.zero_grad()

          label_pred = model(train_img.to(device))
          label_true = train_label.unsqueeze(1).float().to(device)

          train_loss = criterion(label_pred, label_true)

          
          train_loss.backward()
          optimizer.step()

          train_acc = accuracy_classifier(label_pred, label_true)
          train_loss_list.append(float(_get_tensor_value(train_loss)))
          train_acc_list.append(float(train_acc))

      train_loss_epoch = sum(train_loss_list)/len(train_loader)
      train_acc_epoch = sum(train_acc_list)/len(train_loader)

      history['train_loss'].append(train_loss_epoch)
      history['train_acc'].append(train_acc_epoch)


      ## evaluate
      val_loss_list = []
      val_acc_list = []

      model.eval()
      with torch.no_grad():
          for val_img, val_label in valid_loader:
              
              label_pred = model(val_img.to(device))
              label_true = val_label.unsqueeze(1).float().to(device)
              val_loss = criterion(label_pred, label_true)
              
              val_acc = accuracy_classifier(label_pred, label_true)
              val_loss_list.append(float(_get_tensor_value(val_loss)))
              val_acc_list.append(float(val_acc))
      
      val_loss_epoch = sum(val_loss_list)/len(valid_loader)
      val_acc_epoch = sum(val_acc_list)/len(valid_loader)

      history['val_loss'].append(val_loss_epoch)
      history['val_acc'].append(val_acc_epoch)

      print(f"Epoch= {epoch} \t train_loss = {train_loss_epoch:.4f} \t val_loss = {val_loss_epoch:.4f} \t train_acc= {train_acc_epoch:3f} \t val_acc = {val_acc_epoch:.3f}")

      
      if val_acc_epoch > max_acc:
             max_acc = val_acc_epoch
             
             best_model_wts = copy.deepcopy(model.state_dict())
             torch.save(model.state_dict(), os.path.join(args.path_outputs, f"classifier_best.pt"))
             print(f"New best validation accuracy : {max_acc}, model saved")
   ## plot accuracy
   # plt.figure()
   # plt.subplot(121)
   # plt.plot(history['train_loss'], label="Tr Loss")
   # plt.plot(history['val_loss'], label="Val Loss")
   # plt.legend()
   # plt.xlabel('Epoch Number')
   # plt.ylabel('Loss')

   # plt.subplot(122)
   # plt.plot(history['train_acc'], label="Tr Acc")
   # plt.plot(history['val_acc'], label="Val Acc")
   # plt.legend()
   # plt.xlabel('Epoch Number')
   # plt.ylabel('Acc')
   # plt.subplots_adjust(wspace=0.5)
   # plt.show()

   ##-----------------------------------labeling images using the best model 


   model.load_state_dict(best_model_wts)
   model.eval()

   model_preds = []

   with torch.no_grad():
      for image_file in tqdm(os.listdir(args.path_imgs)):
         if image_file.lower().endswith((".tif", ".tiff")):
             # img_names.append(os.path.splitext(image_file)[0])
            image = tifffile.imread(os.path.join(args.path_imgs, image_file))
            image = np.stack((image,)*3, axis=-1) # the model is trained with 3 channels 
            image = img_transform(image).to(device)
         
            output = model(image.unsqueeze(0))
            model_preds.append(output.item())
   df_labels = pd.read_csv(args.path_labels)
   print(f'Number of predicted labels: {len(model_preds)}')
   print(f'Number of rows in the dataframe: {df_labels.shape[0]}')
   if len(model_preds) > df_labels.shape[0]:
       df_labels['preds'] = model_preds[:df_labels.shape[0]]
   else:
       df_labels['preds'] = model_preds
       
       

   ## Classify the images into 3 classes at the end:
   # 0: disconnected
   # 1: in between
   # 2: connected
   # to find the threshold values, save model predictions and plot the histogram
   # best_thresh1 = 0.162 # obtained from histogram to match with manual labels
   best_thresh1 = 0.15
   best_thresh2 = 0.95
   conditions = [
        (df_labels['preds'] < best_thresh1),
        (df_labels['preds'] >= best_thresh1) & (df_labels['preds'] <= best_thresh2),
        (df_labels['preds'] > best_thresh2)
    ]
   choices = [0, 1, 2]

   df_labels['label'] = np.select(conditions, choices)

   df_labels.to_csv(os.path.join(args.path_outputs, 'df_labels.csv'), index= False)


if __name__=='__main__':
   fine_tune_resnet50()