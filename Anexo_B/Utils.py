################################################################################
# Title: reference_wave                                                        #
#                                                                              #
# The function computes the digital reference wave [Eq. (5)]                   #
#                                                                              #                                                                             
# Authors: Raul Castaneda and Ana Doblas                                       #
# Department of Electrical and Computer Engineering, The University of Memphis,# 
# Memphis, TN 38152, USA.                                                      #   
#                                                                              #
# Email: rcstdqnt@memphis.edu and adoblas@memphis                              #
################################################################################

import numpy as np

def reference_wave(M, N, m, n, wavelength, dxy, fx_max, fy_max, k,fx_0,fy_0):
    
    theta_x = np.arcsin((fx_0 - fx_max) * wavelength / (M * dxy))
    theta_y = np.arcsin((fy_0 - fy_max) * wavelength / (N * dxy))
    ref_wave = np.exp(1j * k * (np.sin(theta_x) * m * dxy + np.sin(theta_y) * n * dxy))
    
    return ref_wave
