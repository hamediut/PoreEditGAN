import numpy as np
from numba import jit
from tqdm import tqdm
from typing import Dict, List

@jit 
# --> It is preferred to use numba here for a speed-up, if installed!!
def two_point_correlation(im, dim, var=0):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 0)
    
    The input image im is expected to be three-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[1] #y-axis
        dim_2 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2-r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a+r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a+r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r]/(float(lmax))
    return two_point

def calculate_smd_list(images):
    """This function calculate s2 for a stack of images and return a list of correlations s2 and scaled autocovariance
    The list contains correaltions of all the slices"""
    fn_list = []
    s2_list = []
#     Nr_list = []
    if len(images.shape) == 3: # if you read slices in a one stack of numpy array
        
        for i in tqdm(range(images.shape[0])):
            two_pt_dim0 = two_point_correlation(images[i], dim = 0, var = 1) #S2 in x-direction
            two_pt_dim1 = two_point_correlation(images[i], dim = 1, var = 1) #S2 in y-direction

            #Take average of directions; use half linear size assuming equal dimension sizes
            Nr = two_pt_dim0.shape[0]//2

            S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
            S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
            S2_average = ((S2_x + S2_y)/2)[:Nr]

            s2_list.append(S2_average)

            # autoscaled covriance---------------------------------------
            f_average = (S2_average - S2_average[0]**2)/S2_average[0]/(1 - S2_average[0])
            fn_list.append(f_average)
            
             # N(L) number of different size lines
#             Nr_list.append(Nr)
        return s2_list, fn_list
    
    elif len(images.shape) == 2: # in case we only have a 2D image, don't need to return a list 

        two_pt_dim0 = two_point_correlation(images, dim = 0, var = 1) #S2 in x-direction
        two_pt_dim1 = two_point_correlation(images, dim = 1, var = 1) #S2 in y-direction

        #Take average of directions; use half linear size assuming equal dimension sizes
        Nr = two_pt_dim0.shape[0]//2

        S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
        S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
        S2_average = ((S2_x + S2_y)/2)[:Nr]

        s2_list.append(S2_average)

        # autoscaled covriance---------------------------------------
        f_average = (S2_average - S2_average[0]**2)/S2_average[0]/(1 - S2_average[0])
        fn_list.append(f_average)
        
        # N(L) number of different size lines
#         Nr_list.append(Nr)
        return S2_average, f_average