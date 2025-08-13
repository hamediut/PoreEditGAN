# Microstructure characterisation using SMDs
⚠️ **Work in Progress**: This branch is under active development. Code and documentation are being updated regularly.
git a
This branch contains codes, data, and documentation for **part 1** of the paper *"Transient porosity during fluid-mineral interaction. Part 1: In-situ 4D tomography"*. It involves codes for characterising microstructures using statistical microstructure descriptors (SMDs) and Minkowskie functionals (MFs). Both parts have been recently submitted to JGR: Solid Earth. The link to the papers will be provided when the pre-print versions are out. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

To edit an image, a few steps should be taken in order. Here is the summary of these steps (more details can be found in the our paper):

1) Train an StyleGAN2-ADA with you own images, see the original [Github page](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file).
2) After training your GAN, generate images (the more, the merrier, like few thausands)
3) Invert the generated images (can be along with some real images) to the latent space ($W$ space) of the trained Generator.
Here, we use a hybrid method consisting method consisting training an Encoder and an following optimisation step. During trainin gthe Encoder, the pre-trained generator remain fixed and we tran the Encoder against an Discriminator, following the methodolgy in this [paper](https://arxiv.org/abs/2004.00049), see also their [GitHub page](https://github.com/genforce/idinvert_pytorch). 
4) After inverting the images, a binary decision boundary should be found in the latent space. This binary boundary in our case is between connected and disconnected pores. To find this boundary, we need to label the inverted images (here as connected with label 1, and disconnected labelled as 0). Then, a linear support vector machine (SVM) is used to find the hyperplane in the latent space separating the connected and disconnected regions, following the work [InterFaceGAN](https://genforce.github.io/interfacegan/).
5) The last step, it to push the the inverted code of the disconnected image along the direction orthogonal to the hyperplane to connect the pore network. 



## Usage
Here you can find some guidelines and example to run the scripts for microstructure characterisation using Statistical microstructure descriptors (SMDs) and editing the pores using our trained models.
The codes here are for images of size 512 by 512 pixels. For smaller images (256 by 256), see the branch ``feature-res256``.
### Overview

Here is the summary of scripts in this repository:

- ``SMDs.py``: Computes Statistical microstructure descriptors (SMDs) in each image.
