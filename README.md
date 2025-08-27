⚠️ **Work in Progress**: This branch is under active development. Code and documentation are being updated regularly.

This repository provides codes and documentation for a two-part paper recenty submitted to JGR: Solid Earth. This main branch contains the code for the second part of the paper titled *"Transient porosity during fluid-mineral interaction. Part 2: Generative AI"*. For the part 1 see the branch "paper-part1". The link to the papers will be provided when the pre-print versions are out. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

## PoreEditGAN
This main branch provides codes for editing pores using the latent space of pre-trained GANs. It also contains the code for characterising the pore networks using spatial correlation functions. 

To edit an image, a few steps should be taken in order. Here is the summary of these steps (more details can be found in the our paper):

1) Train an StyleGAN2-ADA with you own images, see the original [Github page](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file).
2) After training your GAN, generate images (the more, the merrier, like few thausands)
3) Invert the generated images (can be along with some real images) to the latent space ($W$ space) of the trained Generator.
Here, we use a hybrid method consists of training an Encoder followed by an optimisation step. During training the Encoder, the pre-trained generator remain fixed while the Encoder is trained against an Discriminator, following the methodolgy in this [paper](https://arxiv.org/abs/2004.00049), see also their [GitHub page](https://github.com/genforce/idinvert_pytorch). 
4) After inverting the images, a binary decision boundary should be found in the latent space. This binary boundary in our case is between connected and disconnected pores. To find this boundary, we need to label the inverted images (here as connected with label 1, and disconnected labelled as 0). Then, a linear support vector machine (SVM) is used to find the hyperplane in the latent space separating the connected and disconnected regions, following the work [InterFaceGAN](https://genforce.github.io/interfacegan/).
5) The last step, it to push the the inverted code of the disconnected image along the direction orthogonal to the hyperplane to connect the pore network. 



## Usage
Fir training the styleGAN2-ADA and the encoder described above see the github pages mentioned above. Here, we assume that those steps are already taken. You can find the pre-trained generator and encoder in our Zenodo page.
<!-- Here you can find some guidelines and example to run the scripts for microstructure characterisation using Statistical microstructure descriptors (SMDs) and editing the pores using our trained models.
The codes here are for images of size 512 by 512 pixels. -->

 <!-- For smaller images (256 by 256), see the branch ``feature-res256``. -->


### Invert images

To invert images into the latent space (W space) of the pre-trained generator, run the following :

**Example:**
```powershell
python invert_imgs.py --path_imgs "Dataset\res512\labelled_images" --path_G "Dataset\res512\TrainedModels_res512\network-snapshot-009404.pkl" --path_E "Dataset\res512\TrainedModels_res512\Encoder_22200.pt" --path_VGG "Dataset\res512\TrainedModels_res512\vgg16.pth" --res 512 --path_output "Dataset\outputs" 
```

The script's output is a dictionary file ('SMDs.pkl') containing the SMDs for each slice (image at different times) in your 3D tif or for only one image if your input image is a single 2D tif image. If multiple images provided, then you will have another dictionary for Omega values, saved as 'omega_SMDs.pkl'. If you run the code with our dataset and you give the path to the timelog, then the script gives a dataframe saved as 'df_SMDs_2D_omega.csv' with all the omega values at different columns, so you can then plot them versus time, as you can see in the jupyter notebook. If you use your own 2D images, you get the results as a dictionary 'omega_SMDs.pkl' in your output path.

**Arguments:**
Here are arguments for the above script:
- `--path_input`: path to the folder of labelled images are, that is, the images you want to invert.
- `--path_G`: full path to the pre-trained generator.
- `--path_E`: full path to the pre-trained encoder.
- `--path_VGG`: full path to the pre-trained VGG model.
- `--res`: image resolution, default = 512
- `--path_output`: path to the output folder to save dictinary containing inverted codes and the results of inversion.

For other arguments see the docstring of the script.


### Semantic boundary
Once the images are inverted and their corresponding latent codes are obtained (see the previous step), the semantic boundary for editing the pore connecivity can be found by running the following:

```powershell
python find_boundary.py --path_log "Dataset\res512\log_all.txt" --path_labels "Dataset\res512\df_labels.csv" --path_latents "Dataset\res512\latent_codes.pkl" --path_output "Dataset\outputs"
```

**Arguments:**
Here are arguments for the above script:
- `--path_log`: full path to the log file created during inverting the images (log.txt).
- `--path_labels`: full path to the labels (df_labels.csv) file in the dataset.
- `--path_latents`: full path to the latent codes obtained from previosu step (latent_codes.pkl)
- `--mse_thresh`: threshold for the mse between s2 of real image and reconstructed images, to only get the latent codes of high quality inversions. 
- `--res`: image resolution, default = 512
- `--path_output`: path to the output folder to save boundary and classifier.

The images for inversion have three labels: 0 for disconnected, 2 for connected, and 1 for the images in between. However, the same number of connected(2) and disconnected (0) labels are used to find the decision boundary using SVM approach.

