# Microstructure characterisation using SMDs
⚠️ **Work in Progress**: This branch is under active development. Code and documentation are being updated regularly.

This branch contains codes, data, and documentation for **part 1** of the paper *"Transient porosity during fluid-mineral interaction. Part 1: In-situ 4D tomography"*. It involves codes for characterising microstructures using statistical microstructure descriptors (SMDs) and Minkowskie functionals (MFs). Both parts have been recently submitted to JGR: Solid Earth. The link to the papers will be provided when the pre-print versions are out. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

## Input data and examples
Here, I provide input data which are the images used in the part 1 of the paper. The raw data are segmented images and can be accessed through Zenodo repository that will be provided soon here.


## Compile cpp code
First step to compute the SMDs is to compile cpp code based on the size of your image(s). The cpp code can be found in this repository in 'cpp_poly\512\Cpp_source\Polytope\Sample_Pn_UU.cpp'. In this script MAXX should be set to your image size +1 and NT to the half of your image size. For instance, if your image(s) are 512 by 512 pixels. MAXX == 513, and NT = 256. 

After this, you should compile the file and remove the .exe from the file name to become 'Sample_Pn_UU'.

## Calculate SMDs
After compiling the cpp code. you can run the following command to compute SMDs for your 2D image(s). The input can be a single 2D tif image or a number of 2D slices stacked as an 3D tif image of shape (num_slice, W, H) as in our case. 

```
python calculate_SMD.py --path_input "C:\Users\David\OneDrive - Universiteit Utrecht\My PhD\My papers\2ndPaper_4Dimages\Part1_chapter
3\Data_ForGithub\Fig04\Fig04a\img_xy_slice_100.tif" --cpathPn "D:\Hamed\PoreEditGAN_github\cpp_poly\512\Cpp_source\Polytope" --runtimePn "D:\Hamed\PoreEditGAN_github\cpp_poly\512\runtime" --outputPn "D:\Hamed\PoreEditGAN_github\cpp_poly\512\runtime\output" --path_output "C:\Users\David\OneDrive - Universiteit Utrecht\My PhD\My papers\2ndPaper_4Dimages\Part1_chapter3\Data_ForGithub\Fig04\Fig04a"
```

### Overview

Here is the summary of scripts in this repository:

