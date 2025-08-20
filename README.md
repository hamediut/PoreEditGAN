# Microstructure characterisation using SMDs
⚠️ **Work in Progress**: This branch is under active development. Code and documentation are being updated regularly.

This branch contains codes, data, and documentation for **part 1** of the paper *"Transient porosity during fluid-mineral interaction. Part 1: In-situ 4D tomography"*. It involves codes for characterising microstructures using statistical microstructure descriptors (SMDs) and Minkowskie functionals (MFs). Both parts have been recently submitted to JGR: Solid Earth. The link to the papers will be provided when the pre-print versions are out. This project is part of my PhD at Utrecht University, funded by European Reseach Councoul (ERC) starting grant 'nanoEARTH' (#852069).

## Overview
- **Goal:** To characterise microstructures using statistical microstructure descriptors (SMDs) and minkowskie functionals.
- **Inputs:** 3D `.tif` stacks or 2D `.tif` slices.
- **Outputs:** Dictionaries `.pkl` file containing the probabilities for each SMD, see the preprint for more details (link will be provided upon release). The jupyter notebook `polytopes_example.ipynb` shows an example of how to plot the SMDs from the output dictionary file.

## Environment setup
You can create the Conda environment with any name you like (e.g., smd_env), then install all packages listed in `environment.yml`:

```
conda env create -f environment.yml -n smd_env
conda activate smd_env
```

## Input data and examples
Here, I provide input data which are the images used in the part 1 of the paper. The raw data are segmented images and can be accessed through Zenodo repository that will be provided soon here.


## Compile cpp code
First step to compute the SMDs is to compile cpp code based on the size of your image(s). The cpp code can be found in this repository in 'cpp_poly\512\Cpp_source\Polytope\Sample_Pn_UU.cpp'. In this script MAXX should be set to your image size +1 and NT to the half of your image size. For instance, if your image(s) are 512 by 512 pixels, then in cpp code you should set #define MAXX 513, and #define Nt 256. 

After this, you should compile the file and remove the .exe from the file name to become 'Sample_Pn_UU'.

## Usage

### Calculate SMDs
After compiling the cpp code. you can run the following command to compute SMDs for your 2D image(s). The input can be a single 2D tif image or a number of 2D slices stacked as an 3D tif image of shape (num_slice, W, H) as in our case. 

**Example:**
```powershell
python calculate_SMD.py --path_input "Dataset\imgs_512\img_xy_slice_100.tif" --cpathPn "D:\Hamed\PoreEditGAN_github\cpp_poly\512\Cpp_source\Polytope" --runtimePn "D:\Hamed\PoreEditGAN_github
\cpp_poly\512\runtime" --outputPn "D:\Hamed\PoreEditGAN_github\cpp_poly\512\runtime\output" --path_output "Results\imgs_512"
```
**Arguments:**
Here are arguments that should be passed when running the above code:
- `--path_input`: full path tp your tif image.
- `--cpathPn`: path to Polytope folder in `cpp_poly\512\Cpp_source\Polytope`.
- `--runtimePn`: path to runtime folder in `cpp_poly\512\runtime`.
- `--outputPn`: path to output folder in runtime `cpp_poly\512\runtime\output`
- `--path_output`: path to the output folder to save dictinary containing the SMDs as a `.pkl` file.

### S2 for a single 3D image
To calculate the S_2 and F_2 for a single 3D image, run:
```powershell
python calculate_S2_3D.py --path_input "D:\Hamed\PSI\Stacks3D\exp1\Binary_KBr07_0140_f815_t1080_coordinates_541,575_siz
e_512.tif" --path_output "C:\path\to\output"
```
**Arguments:**
Here are arguments that should be passed when running the above code:
- `--path_input`: full path to your 3D tif image.
- `--path_output`: path to the output folder to save dictinary containing the s2 in different directions and the radial one as a `s2_3D.pkl` file.

### S2 for 4D images and Omega metric
You can also compute S2 and F2 for a time-resolved 3D stacks (4D images) if you have saved them with sequential file names in a folder. For example, see the folder "Stacks3D\exp1" in the published dataset for exp1 in the paper. When you have your images properly in the folder, you can then run:
```powershell
python calculate_s2_4D.py --path_folder "D:\Hamed\PSI\Stacks3D\exp1" --path_timelog "D:\Hamed\PSI\TimeLogs\timeseries_exp1.log"--path_output "C:\path\to\output"
```
**Arguments:**
Here are arguments that should be passed when running the above code:
- `--path_folder`: path to the folder where you have a bunch of 3D tif images acquired at different times (4D dataset).
- `--path_output`: path to the output folder to save the outputs.

Outputs are two pickle files 's2_3D_dic_r.pkl' and 'f2_3D_dic_r.pkl' whose keys are the experiments'names and stack number. For instance, in our case 'KBr07_0001' means stack number 1 from experiment KBr07. The code takes the experiment name from what is between the first and second '_' in the file name (e.g., Binary_KBr07_0001_f815_t1080_coordinates_541,575_size_512.tif). So if you give smilar names to your images, you don't need to change the code. Otherwise, please modify the code to deal with that. The third output is a csv file 'df_{exp_nam}.csv', containing the time in out experiments and Omega metrics calculated from S2 and f2 functions in 3D that are in the pickle files. With Omega values and delta_omega values, you can then plot the evolution of microstructures as in Figure 8 in our paper (see also the jupyter notebook). 





