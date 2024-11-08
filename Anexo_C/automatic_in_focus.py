#############################################################
# Title: Automatic focussing reconstruction                                                                                                    
#                                                                             
# Description: This algorithm allows obtain the best porpagation distance      
# for holograms recorded in out-of-focus  setup                                
#                                                                              
# Authors: Raul Castaneda, Emmanuel Mazo                                                      
# Applied optics group                  
# EAFIT university                                            
# Medellín, Colombia.                                                             
#                                                                              
# Email: andrescatanedaiff@gmailc.com                                        
# version 1.0 (2023)                    
#############################################################

# Required libraries for numerical processing, optimization, and visualization
from scipy.optimize import minimize  # For optimization of focusing parameters
import numpy as np                  # For numerical computations
import matplotlib.pyplot as plt     # For visualization            # Custom module for reference wave generation
import Utils    # Custom module for optimization criteria

def Plot1Imagen(original, title):
    """
    Display a single image with a title.
    
    Args:
        original: Image array to display
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(original, cmap='gray')
    plt.title(title)
    plt.show()

def saveimage(image, name):
    """
    Save an image to disk with specified name.
    
    Args:
        image: Image array to save
        name: Filename for saving
    """
    plt.imsave(name, image, cmap='gray')
    print(str(name) + "Guardada")

def rec(holo, lamb, dxy, radius, k, fx_0, fy_0):
    """
    Reconstruct a hologram using automatic focusing techniques.
    
    This function implements an automatic focusing algorithm for digital hologram 
    reconstruction. It finds the optimal propagation distance for out-of-focus 
    holograms using optimization techniques.
    
    Args:
        holo: Input hologram array
        lamb: Wavelength of light used (in consistent units)
        dxy: Pixel size (sampling interval)
        radius: Radius for spatial frequency filtering
        k: Wave number (2π/λ)
        fx_0, fy_0: Central frequencies in x and y directions
    
    Returns:
        field_compensate: Reconstructed and compensated complex field
    """
    # Get hologram dimensions and create coordinate grid
    M, N = holo.shape
    n, m = np.meshgrid(np.arange(-N//2, N//2), np.arange(-M//2, M//2))

    # Apply spatial frequency filtering to isolate the relevant orders
    holo_filter, fx_max, fy_max = Utils.filter_hologram(holo, M, N, radius)
    #print(fx_max, fy_max)

    # Initialize optimization with detected peak positions
    # The peaks correspond to the carrier frequencies of the hologram
    seed_maxPeak = [np.float64(fx_max), np.float64(fy_max)]

    # Configure optimization parameters
    options = {
        'disp': False,     # Suppress optimization display
        'maxiter': 100     # Maximum number of iterations
    }

    # Initial evaluation of the cost function
    J = Utils.minimization_compensation(
        seed_maxPeak, lamb, dxy, M, N, m, n, holo_filter, k, fx_0, fy_0
    )

    # Perform optimization to find best focusing parameters
    # Uses scipy's minimize function with the compensation criterion
    res = minimize(
        lambda t: Utils.minimization_compensation(
            t, lamb, dxy, M, N, m, n, holo_filter, k, fx_0, fy_0
        ),
        seed_maxPeak,
        tol=1e-6,         # Tolerance for convergence
        options=options
    )

    # Extract optimized parameters
    MaxPeaks = res.x      # Optimal peak positions
    J = res.fun           # Final value of cost function
    
    # Extract optimized frequency components
    fx_max_temp = MaxPeaks[1]
    fy_max_temp = MaxPeaks[0]

    # Create reference wave using optimized parameters
    ref_wave = Utils.reference_wave(
        M, N, m, n, lamb, dxy, fx_max_temp, fy_max_temp, k, fx_0, fy_0
    )

    # Compensate the filtered hologram with the optimized reference wave
    field_compensate = ref_wave * holo_filter

    return field_compensate