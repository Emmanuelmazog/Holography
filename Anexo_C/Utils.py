################################################################################
# Title: filter_holgram                                                         #
#                                                                              #
# The function is used to filter the +1 order of the hologram                  #
#                                                                              #                                                                           
# Authors: Raul Castaneda,    Ana Doblas                                       #
# Department of Electrical and Computer Engineering, The University of Memphis, #
# Memphis, TN 38152, USA.                                                      #   
#                                                                              #
# Email: adoblas@memphis.edu                                                   #
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Plot1Imagen(original, title):
    """
    Display a single image with title.
    
    Args:
        original: Image array to display
        title: Title for the plot
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(original, cmap='gray')
    plt.title(title)
    plt.show()

def filter_hologram(holo, M, N, radius):
    """
    Filter the +1 diffraction order from a hologram's Fourier spectrum.
    
    This function implements spatial filtering in the frequency domain to isolate
    the +1 diffraction order of a digital hologram. The process involves:
    1. Computing the Fourier transform
    2. Applying a mask to select the right half of the spectrum
    3. Finding the maximum peak (corresponding to +1 order)
    4. Applying a circular filter around this peak
    
    Args:
        holo: Input hologram array
        M, N: Dimensions of the hologram
        radius: Radius of the circular filter
    
    Returns:
        tuple: (
            filtered_hologram: Complex array of filtered hologram,
            fx_max: x-coordinate of +1 order peak,
            fy_max: y-coordinate of +1 order peak
        )
    """
    # Compute centered Fourier transform of the hologram
    # 1. ifftshift centers the hologram for FFT
    # 2. fft2 computes the 2D FFT
    # 3. fftshift centers the spectrum
    fft_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo)))
    
    # Create initial mask to select right half of spectrum
    # This removes the -1 order and DC term
    mask = np.ones((M, N))
    mask[:, 0 : round(M/2) + 10] = 0  # Zero out left half plus margin
    
    # Apply mask and find location of maximum peak
    # This peak corresponds to the center of the +1 order
    fft_holo_I = fft_holo * mask
    maxValue_1 = np.max(np.abs(fft_holo_I))
    fy_max_1, fx_max_1 = np.where(np.abs(fft_holo_I) == maxValue_1)
    fx_max = fx_max_1
    fy_max = fy_max_1
    
    # Create circular filter centered on the +1 order peak
    resc = radius  # Radius of circular filter
    filter1 = np.ones((M, N))
    for r in range(M):
        for p in range(N):
            # Calculate distance from each point to peak
            if np.sqrt((r-fy_max)**2 + (p-fx_max)**2) > resc:
                filter1[r, p] = 0  # Zero out points outside radius
    
    # Apply circular filter to Fourier spectrum
    fft_filter_holo = fft_holo * filter1
    
    # Inverse Fourier transform to get filtered hologram
    # 1. fftshift centers the spectrum for IFFT
    # 2. ifft2 computes the 2D IFFT
    # 3. ifftshift centers the resulting hologram
    out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fft_filter_holo)))

    return out, fx_max, fy_max

def reference_wave(M, N, m, n, wavelength, dxy, fx_max, fy_max, k,fx_0,fy_0):
    
    theta_x = np.arcsin((fx_0 - fx_max) * wavelength / (M * dxy))
    theta_y = np.arcsin((fy_0 - fy_max) * wavelength / (N * dxy))
    ref_wave = np.exp(1j * k * (np.sin(theta_x) * m * dxy + np.sin(theta_y) * n * dxy))
    
    return ref_wave
def minimization_compensation(seed_maxPeak, wavelength, dxy, M, N, m, n, holo_filt, k, fx_0, fy_0):
    """
    Compute the cost function for hologram reconstruction optimization.
    
    This function implements a cost function that evaluates the quality of hologram 
    reconstruction based on the number of binary transitions in the phase distribution.
    A lower cost indicates better focusing.
    
    The algorithm:
    1. Generates a reference wave using current parameters
    2. Reconstructs the hologram
    3. Extracts and normalizes the phase
    4. Binarizes the phase
    5. Counts non-zero pixels as a measure of phase discontinuities
    
    Args:
        seed_maxPeak: Array [fx_max, fy_max] with current peak coordinates
        wavelength: Light wavelength used
        dxy: Pixel size (sampling interval)
        M, N: Dimensions of the hologram
        m, n: Coordinate grids
        holo_filt: Filtered hologram
        k: Wave number (2π/λ)
        fx_0, fy_0: Central frequencies
    
    Returns:
        float: Cost function value (lower is better)
    """
    # Initialize cost function value
    J = 0
    
    # Extract current peak coordinates
    fx_max = seed_maxPeak[0]  # x-coordinate of peak
    fy_max = seed_maxPeak[1]  # y-coordinate of peak
    
    # Generate reference wave using current parameters
    ref_wave = reference_wave(
        M, N, m, n, wavelength, dxy, fx_max, fy_max, k, fx_0, fy_0
    )
    
    # Reconstruct hologram by multiplying with reference wave
    holo_rec = holo_filt * ref_wave
    
    # Extract phase information
    phase = np.angle(holo_rec)
    
    # Normalize phase to range [0, 1]
    phase_normalized = cv2.normalize(
        phase, None, 
        alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_32F
    )

    # Convert to 8-bit grayscale (0-255)
    phase_save = np.uint8(255 * phase_normalized)

    # Binarize the image with threshold 25 (≈0.1 * 255)
    # This separates the phase into two regions
    _, ib = cv2.threshold(phase_save, 25, 255, cv2.THRESH_BINARY)

    # Compute cost as number of dark pixels
    # Lower number of dark pixels indicates better focusing
    J = M * N - np.sum(ib > 0)

    return J
def filter_hologram(holo, M, N, radius):
    """
    Filter the +1 diffraction order from a hologram's Fourier spectrum.
    
    This function implements spatial filtering in the frequency domain to isolate
    the +1 diffraction order of a digital hologram. The process involves:
    1. Computing the Fourier transform
    2. Applying a mask to select the right half of the spectrum
    3. Finding the maximum peak (corresponding to +1 order)
    4. Applying a circular filter around this peak
    
    Args:
        holo: Input hologram array
        M, N: Dimensions of the hologram
        radius: Radius of the circular filter
    
    Returns:
        tuple: (
            filtered_hologram: Complex array of filtered hologram,
            fx_max: x-coordinate of +1 order peak,
            fy_max: y-coordinate of +1 order peak
        )
    """
    # Compute centered Fourier transform of the hologram
    # 1. ifftshift centers the hologram for FFT
    # 2. fft2 computes the 2D FFT
    # 3. fftshift centers the spectrum
    fft_holo = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo)))
    
    # Create initial mask to select right half of spectrum
    # This removes the -1 order and DC term
    mask = np.ones((M, N))
    mask[:, 0 : round(M/2) + 10] = 0  # Zero out left half plus margin
    
    # Apply mask and find location of maximum peak
    # This peak corresponds to the center of the +1 order
    fft_holo_I = fft_holo * mask
    maxValue_1 = np.max(np.abs(fft_holo_I))
    fy_max_1, fx_max_1 = np.where(np.abs(fft_holo_I) == maxValue_1)
    fx_max = fx_max_1
    fy_max = fy_max_1
    
    # Create circular filter centered on the +1 order peak
    resc = radius  # Radius of circular filter
    filter1 = np.ones((M, N))
    for r in range(M):
        for p in range(N):
            # Calculate distance from each point to peak
            if np.sqrt((r-fy_max)**2 + (p-fx_max)**2) > resc:
                filter1[r, p] = 0  # Zero out points outside radius
    
    # Apply circular filter to Fourier spectrum
    fft_filter_holo = fft_holo * filter1
    
    # Inverse Fourier transform to get filtered hologram
    # 1. fftshift centers the spectrum for IFFT
    # 2. ifft2 computes the 2D IFFT
    # 3. ifftshift centers the resulting hologram
    out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fft_filter_holo)))

    return out, fx_max, fy_max