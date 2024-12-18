## PoreEditGAN
This repository provides codes for editing pores using the latent space of pre-trained GANs. It also contains the code for characterising the pore networks using spatial correlation functions. There are two papers related to this Github page which has not been yet submitted. The link to the papers will be provided when the pre-print versions are out. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

To edit an image, a few steps should be taken in order. Here is the summary of these steps (more details can be found in the our paper):

1) Train an StyleGAN2-ADA with you own images, see the original [Github page](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file).
2) After training your GAN, generate images (the more, the merrier, like few thausands)
3) Invert the generated images (can be along with some real images) to the latent space ($W$ space) of the trained Generator.
Here, we use a hybrid method consisting method consisting training an Encoder and an following optimisation step. During trainin gthe Encoder, the pre-trained generator remain fixed and we tran the Encoder against an Discriminator, following the methodolgy in this [paper](https://arxiv.org/abs/2004.00049), see also their [GitHub page](https://github.com/genforce/idinvert_pytorch). 
4) After inverting the images, a binary decision boundary should be found in the latent space. This binary boundary in our case is between connected and disconnected pores. To find this boundary, we need to label the inverted images (here as connected with label 1, and disconnected labelled as 0). Then, a linear support vector machine (SVM) is used to find the hyperplane in the latent space separating the connected and disconnected regions, following the work [InterFaceGAN](https://genforce.github.io/interfacegan/).
5) The last step, it to push the the inverted code of the disconnected image along the direction orthogonal to the hyperplane to connect the pore network. 



## Usage
### Overview

There several scripts that need to be run in order for editing an image. Here is a summary of scripts.

- ``SMDs.py``: Computes Statistical microstructure descriptors (SMDs) in each image.
