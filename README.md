# PoreEditGAN
This repository provides codes and documentation for a two-part paper recenty submitted to JGR: Solid Earth. This main branch contains the code for the second part of the paper titled *"Transient porosity during fluid-mineral interaction. Part 2: Generative AI"*. For the part 1 see the branch "paper-part1". [Here](https://doi.org/10.22541/essoar.175587740.05926718/v1) you can read the preprint. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

## Overview
This main branch provides codes for editing pores using the latent space of pre-trained GANs. See the branch for the first part ("paper-part1") for characterising the pore space using SMDs. 

To edit an image, a few steps should be taken in order. Here is the summary of these steps (more details can be found in our paper):

1) Train an StyleGAN2-ADA with your own images, see the original [Github page](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file).
2) After training your GAN, generate images (the more, the merrier, like few thausands).
3) Invert the generated images (can be along with some real images) to the latent space ($W$ space) of the trained Generator.
Here, we use a hybrid method consists of training an encoder followed by an optimisation step. During training the encoder, the pre-trained generator remains fixed while the encoder is trained against a discriminator, following the methodolgy in this [paper](https://arxiv.org/abs/2004.00049), see also their [GitHub page](https://github.com/genforce/idinvert_pytorch). 
4) After inverting the images, a binary decision boundary should be found in the latent space. This binary boundary in our case is between connected and disconnected pores. To find this boundary, we need to label the inverted images (here as connected with label 1, and disconnected labelled as 0). Then, a linear support vector machine (SVM) is used to find the hyperplane in the latent space separating the connected and disconnected regions, following the work [InterFaceGAN](https://genforce.github.io/interfacegan/).
5) The last step, it to push the the inverted code of the disconnected image along the direction orthogonal to the hyperplane to connect the pore network. 

## Environment setup
You can create the Conda environment with any name you like (e.g., smd_env), then install all packages listed in `environment.yml`:

```
conda env create -f environment.yml -n smd_env
conda activate smd_env
```

## Data availability
The dataset for this project consists of segmented images and timelog data for the experiments are available on Zenodo: 

## Usage
For training the styleGAN2-ADA and the encoder, see the github pages mentioned above. Here, we assume that those steps are already taken. However, you can find the pre-trained generator and encoder in our Zenodo page.

 For smaller images (256 by 256), see the branch ``feature-res256``.


### Invert images

To invert images into the latent space (W space) of the pre-trained generator, run the following :

**Example:**
```powershell
python invert_imgs.py --path_imgs "Dataset\res512\labelled_images" --path_G "Dataset\res512\TrainedModels_res512\network-snapshot-009404.pkl" --path_E "Dataset\res512\TrainedModels_res512\Encoder_22200.pt" --path_VGG "Dataset\res512\TrainedModels_res512\vgg16.pth" --res 512 --path_output "Dataset\outputs" 
```
The output is a dictionary file ('latent_codes.pkl') whose keys are the image file name and its keys are the latent codes corresponding to each image. 

**Arguments:**
Here are arguments for the above script:
- `--path_input`: path to the folder of labelled images, that is, the images you want to invert.
- `--path_G`: full path to the pre-trained generator.
- `--path_E`: full path to the pre-trained encoder.
- `--path_VGG`: full path to the pre-trained VGG model.
- `--res`: image resolution, default = 512
- `--path_output`: path to the output folder to save dictinary containing inverted codes. The script also outputs the different loss values during the training and export it as log.txt file which is used in the next step. Also the reconstructed image after inverting the image are also are saved in this folder. For other arguments see the docstring of the script.


### Semantic boundary
Once the images are inverted and their corresponding latent codes are obtained (see the previous step), the semantic boundary for editing the pore connecivity can be found by running the following:

```powershell
python find_boundary.py --path_log "Dataset\res512\log.txt" --path_labels "Dataset\res512\df_labels.csv" --path_latents "Dataset\res512\latent_codes.pkl" --path_output "Dataset\outputs"
```

**Arguments:**
Here are arguments for the above script:
- `--path_log`: full path to the log file created during inverting the images (log.txt).
- `--path_labels`: full path to the labels (df_labels.csv) file in the dataset.
- `--path_latents`: full path to the latent codes obtained from previosu step (latent_codes.pkl).
- `--mse_thresh`: threshold for the mse between s2 of real image and reconstructed images, to only get the latent codes of high quality inversions. 
- `--res`: image resolution, default = 512.
- `--path_output`: path to the output folder to save boundary and classifier as pickle files.

In the df_labels.csv, you can see that there is three labels for images: 0 for disconnected, 2 for connected, and 1 for the images in between. However, the same number of connected(2) and disconnected (0) labels are selected and used to find the decision boundary using SVM approach when running the function 'train_boundary'.

### Edit image
After inverting the images and finding the decision boundary in the latent space, then an image can be edited using pre-trained generator and the outputs of previous steps by running the following:

**Example:**
```powershell
python invert_imgs.py --path_latents "Dataset\res512\latent_codes.pkl" --path_boundary "Dataset\res512\boundary.pkl" --path_classifier "Dataset\res512\classifier.pkl"--path_G "Dataset\res512\TrainedModels_res512\network-snapshot-009404.pkl" --path_img "Dataset\imgs_512\Exp06_07_0036.tif" --path_output "Dataset\outputs"
```

**Arguments:**
Here are arguments for the above script:
- `--path_latents`: full path to the latent codes obtained from inverting image (latent_codes.pkl).
- `--path_boundary`: full path to the boundary found in the previous step (*.pkl).
- `--path_classifier`: full path to the trained SVM classifier found in the previous step (*.pkl).
- `--path_G`: full path to the pre-trained generator (network-snapshot-009404.pkl for res= 512).
- `--path_img`: full path to the image you want to edit (tif or png file).
- `--path_output`: path to the output folder to save the edited images.