#############################################################
# Title: Hologram Simulation Program                               
#                                                                              
# Description: This program simulates digital holograms with and without 
# speckle noise, using both realistic and non-realistic approaches                               
#                                                                              
# Authors: Emmanuel Mazo Gómez, Raul Andres Castañeda
# Applied optics group
# EAFIT university                   
#                                           
# Medellín, Colombia.                                                            
#                                                                              
# Email: emazog@eafit.edu.co, racastaneq@eafit.edu.co
#############################################################

#Libraries
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import random
import os
import Utils
from skimage.restoration import unwrap_phase

# Initialize a random seed for reproducibility
np.random.seed(42)
random.seed(42)

#Parameters
lambda_ = 0.633        # source's wavelength in microns
k = 2 * np.pi / lambda_#Wave number
dxy = 3.8              # pixel size along vertical and horizontal direction
M = 512                # dimension camara in pixels (axe X)
N = 512                # dimension camara in pixels (axe y)


def coordenadas(M, N):
    """
    Generate real and Fourier domain coordinates for the hologram simulation.
    
    Args:
        M, N: Dimensions of the coordinate grid
    Returns:
        m, n: Real space coordinate grids
        Fx, Fy: Fourier space coordinate grids
    """
    # Generate real coordinates
    x = np.arange(round(-M/2), round(M/2))  # Create an array from -M/2 to M/2
    y = np.arange(round(-N/2), round(N/2))  # Create an array from -N/2 to N/2
    [m, n] = np.meshgrid(x, y)  # Create a meshgrid from x and y coordinates

    # Fourier domain coordinates
    dFx = 1 / (dxy * M)  # Frequency domain sampling interval in x direction
    dFy = 1 / (dxy * N)  # Frequency domain sampling interval in y direction
    m1 = np.arange(1, M)  # Create an array from 1 to M-1
    m2 = np.arange(1, N)  # Create an array from 1 to N-1
    m1 = m1 - M / 2  # Shift array by M/2 to center it
    m2 = m2 - N / 2  # Shift array by N/2 to center it
    fx = dFx * m1  # Calculate frequency coordinates in x direction
    fy = dFy * m2  # Calculate frequency coordinates in y direction
    [Fx, Fy] = np.meshgrid(fx, fy)  # Create a meshgrid from fx and fy coordinates
    
    return m, n, Fx, Fy  # Return real and Fourier domain coordinates

def Load_Image(name, sizex, sizey):
    """
    Load and resize an image for hologram simulation.
    
    Args:
        name: Path to the image file
        sizex, sizey: Target dimensions for the resized image
    Returns:
        Resized image and its coordinates
    """
    # Load the original image in grayscale
    sample = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    # Resize the image using OpenCV
    imagen_redimensionada = cv2.resize(sample, (sizex, sizey), interpolation=cv2.INTER_LINEAR)

    # Get the dimensions of the resized image
    M, N = imagen_redimensionada.shape

    # Create a coordinate mesh
    m, n = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))

    # Return the resized image, its dimensions, and the coordinate meshes
    return imagen_redimensionada, M, N, m, n

def PlotImagen(original, bina, title1, title2):
    """
    Display two images side by side with titles.
    
    Args:
        original: First image to display
        bina: Second image to display
        title1, title2: Titles for the images
    """
    # Create a figure with a specified size
    plt.figure(figsize=(10, 5))

    # Subplot for the original image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(original, cmap='gray')  # Display the original image in grayscale
    plt.title(title1)  # Set the title for the original image

    # Subplot for the binary image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(bina, cmap='gray')  # Display the binary image in grayscale
    plt.title(title2)  # Set the title for the binary image

    # Show the figure with the subplots
    plt.show()


def Plot1Imagen(original, title, cmap='gray'):
    """
    Display a single image with a colorbar.
    
    Args:
        original: Image to display
        title: Title for the plot
        cmap: Colormap to use
    """
    # Create a figure with a specified size
    plt.figure(figsize=(10, 5))
    # Display the image with the specified colormap
    plt.imshow(original, cmap=cmap)
    # Set the title of the plot
    plt.title(title)
    # Add a colorbar to the plot
    cbar = plt.colorbar()
    # Set the label for the colorbar
    cbar.set_label('Intensidad')
    # Show the plot
    plt.show()

def mask(M, N, radius = M/10):
    """
    Create a circular mask for filtering.
    
    Args:
        M, N: Dimensions of the mask
        radius: Radius of the circular mask
    Returns:
        2D numpy array representing the mask
    """
    # Round the radius value to the nearest integer
    resc = round(radius)  # pixels
    # Initialize a mask with ones
    mask = np.ones((N, N))
    # Loop over each pixel in the mask
    for r in np.arange(1, N):
        for p in np.arange(1, N):
            # Calculate the distance from the center of the mask
            if math.sqrt((r - N/2)**2 + (p - N/2)**2) > resc:
                # Set mask value to 0 if outside the radius
                mask[r, p] = 0
    # Return the mask
    return mask

def Contrast(image, alpha=1, beta=0):
    """
    Adjust the contrast of an image.
    
    Args:
        image: Input image
        alpha: Contrast control parameter
        beta: Brightness control parameter
    Returns:
        Contrast-adjusted image
    """
    # Invert the colors of the image
    imagen_invertida = cv2.bitwise_not(image)
    # Adjust the contrast of the inverted image
    # alpha controls the contrast (default is 1.5, where >1 increases contrast)
    # beta controls the brightness (default is 0, where positive values increase brightness)
    imagen_contraste_ajustado = cv2.convertScaleAbs(imagen_invertida, alpha=alpha, beta=beta)
    # Return the contrast-adjusted image
    return imagen_contraste_ajustado

def scale2(image, Phase_factor=True, factor=1):
    """
    Normalize and scale image values either to phase range [-π, π] or custom range [0, factor].
    
    Args:
        image: Input image to be normalized
        Phase_factor: If True, normalize to [-π, π], if False normalize to [0, factor]
        factor: Maximum value for scaling when Phase_factor is False
    Returns:
        tuple: (original image, normalized image)
    """
    if Phase_factor:
        # Normalize the image to the range [0, 1]
        Max = np.max(image)  # Find the maximum value in the image
        normalize = image / Max  # Scale the image values to [0, 1]
        normalize = (normalize - np.min(normalize)) / (np.max(normalize) - np.min(normalize))  # Normalize to [0, 1]
        # Scale to the range [-π, π]
        normalize = normalize * (2 * np.pi) - np.pi
        # Optionally unwrap phase discontinuities (commented out)
        # normalize = np.unwrap(normalize, discont=np.pi/2)
    else:
        # Normalize the image to the range [0, factor]
        Max = np.max(image)  # Find the maximum value in the image
        normalize = image / Max  # Scale the image values to [0, 1]
        normalize = ((normalize - np.min(normalize)) / (np.max(normalize) - np.min(normalize))) * factor  # Scale to [0, factor]
        # Optionally unwrap phase discontinuities (commented out)
        # normalize = np.unwrap(normalize, discont=2*np.pi)

    return image, normalize

def aleatorio(M, N, radius):
    """
    Generate random coordinates for reference wave placement within specific boundaries.
    
    Args:
        M, N: Image dimensions
        radius: Radius constraint for coordinate generation
    Returns:
        tuple: (x, y) coordinates
    """
    # Randomly choose a boundary condition (0 or 1)
    boundry = random.randint(0, 1)
    if boundry == 0:
        # For boundary 0, generate a random y-coordinate within a certain range
        y = random.randint(round(radius), round(N/2) - round(radius + 20)) + 0.25
        # Calculate the corresponding x-coordinate based on a circle equation
        x = round(math.sqrt(abs((((3)**2) * (radius)**2) - (y - N/2)**2)) + (M/2)) + 0.25
    else:
        # For boundary 1, generate a random y-coordinate within a different range
        y = random.randint(round(N/2) + round(radius + 20), N - round(radius)) + 0.25
        # Calculate the corresponding x-coordinate based on a circle equation
        x = round(math.sqrt(abs((((3)**2) * (radius)**2) - (y - N/2)**2)) + (M/2)) + 0.25
    # Return the randomly generated x and y coordinates
    return x, y

def Random_Coordinates(width, height, radius):
    """
    Generate random coordinates along diagonal corners with specific constraints.
    
    Args:
        width, height: Image dimensions
        radius: Radius constraint for coordinate generation
    Returns:
        tuple: (x, y) coordinates
    """
    p = 60  # Padding value for coordinate generation
    # Randomly choose a corner: 0 for top-right, 1 for bottom-left
    corner = random.choice([0, 1])
    # Generate an angle of 45 degrees (pi/4) along the diagonal
    theta = random.choice([math.pi / 4, 7 * math.pi / 4]) 
    # Calculate a random distance in the range [2*radius - p, 2*radius]
    R = random.uniform(2 * radius - p, 2 * radius-15)
    # Calculate the relative coordinates from the origin (0,0)
    x_rel = R * math.cos(theta)
    y_rel = R * math.sin(theta)
    
    if corner == 0:  # top-right corner
        x = width - x_rel
        y = y_rel
    else:  # bottom-left corner
        x = x_rel
        y = height - y_rel
        
    # Randomly decide if the coordinates will be integers or floats with 2 decimal places
    if random.choice([True, False]):
        x = round(x)
        y = round(y)
    else:
        x = round(x, 2)
        y = round(y, 2)

    return (x, y)

def Speckle(sample):    
    """
    Add speckle noise to an image.
    
    Args:
        sample: Input image
    Returns:
        Image with added speckle noise
    """
    # Get the dimensions of the input image
    x, y = sample.shape
    
    # Generate a random scale factor for the speckle noise
    scale = np.random.randint(1, 6) / 50
    print("scale: ", scale)
    # Create speckle noise with a normal distribution
    # loc=np.pi sets the mean of the distribution
    # scale sets the standard deviation of the distribution
    # size=(x, y) generates a noise matrix of the same size as the input image
    Speckle = np.random.normal(loc=np.pi, scale=scale, size=(x, y))
    
    # Add the speckle noise to the original image
    ImageSpeckle = sample + Speckle
    # Return the image with added speckle noise
    return ImageSpeckle

def save_image(array, path):
    """
    Save a numpy array as an image file.
    
    Args:
        array: Numpy array containing image data
        path: Output file path
    """
    plt.imsave(path, array, cmap='gray')

def Not_R_Hologram_simulation(name):
    # Load and preprocess the image
    Sample, M, N, m, n = Load_Image(name, 768, 768)
    Invert_Sample = Contrast(Sample)
    
    # Plot the original and contrast-adjusted images
    #Plot1Imagen(Sample, "Sample")
    #Plot1Imagen(Invert_Sample, "Invert_Sample")
    
    # Normalize the inverted image and add speckle noise
    _a, normalize = scale2(Invert_Sample, Phase_factor=False, factor=2*np.pi)
    sample_Speckle = Speckle(normalize)
    #print(sample_Speckle.shape)
    _a , sample_Speckle1 = scale2(sample_Speckle, Phase_factor=False, factor=2*np.pi)
    #print(sample_Speckle1.shape)
    # Plot the normalized sample and the sample with speckle noise
    #PlotImagen(sample_Speckle1, normalize, "Sample Speckle", "Normalize sample")

    # Define the center of the image
    fx_0 = M / 2
    fy_0 = N / 2
    
    # Generate coordinate grids for the Fourier domain
    m, n, Fx, Fy = coordenadas(M, N)
    
    # Generate a random position for the reference wave
    x, y = aleatorio(M, N, radius=M/10)
    fx_max = x
    fy_max = y
    
    # Calculate the angles for the reference wave
    theta_x = np.arcsin((fx_0 - fx_max) * lambda_ / (M * dxy))
    theta_y = np.arcsin((fy_0 - fy_max) * lambda_ / (N * dxy))
    
    # Create the reference wave
    ref_wave = np.exp(1j * k * (np.sin(theta_x) * m * dxy + np.sin(theta_y) * n * dxy))
    
    # Generate the mask for filtering
    mas = mask(768, 768, radius=M/10)
    
    # Create the object wave without and with speckle noise
    objectWOutS = np.exp(1j * normalize)
    objectWS = np.exp(1j * sample_Speckle)
    
    # Compute the Fourier Transforms of the object waves
    FTobjWOutS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectWOutS)))
    FTobjWS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectWS)))


    # copute magnitud and aply log scale
    magnitude_spectrum_WOutS = np.abs(FTobjWOutS)
    magnitude_spectrum_log_WOutS = np.log1p(magnitude_spectrum_WOutS)  # log1p to avoid log(0)
    # copute magnitud and aply log scale
    magnitude_spectrum_WS = np.abs(FTobjWS)
    magnitude_spectrum_log_WS = np.log1p(magnitude_spectrum_WS)  # log1p to avoid log(0)

    #PlotImagen(magnitude_spectrum_log_WOutS, magnitude_spectrum_log_WS, "FT without Speckle", "FT with Speckle")
    
    # Apply the mask and perform inverse Fourier Transform
    objFiltWOutS_NONR = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FTobjWOutS * mas)))
    objFiltWS_NONR = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FTobjWS * mas)))
    
    # Generate the holograms
    hologramWOutS_NONR = np.abs(ref_wave + objFiltWOutS_NONR)**2
    hologramWS_NONR = np.abs(ref_wave + objFiltWS_NONR)**2
    # Save hologramWS_NONR as an image
    #plt.imsave('hologramWS_NONR.png', hologramWS_NONR, cmap='gray')
    #PlotImagen(hologramWOutS_NONR, hologramWS_NONR, "Hologram WOutS NOTR", "Hologram WS NOTR")

    # Compute the Fourier Transforms of the object waves
    FThologram_WOutS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologramWOutS_NONR)))
    FThologram_WS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologramWS_NONR)))
    # copute magnitud and aply log scale
    magnitude_spectrum_WOutS = np.abs(FThologram_WOutS)
    magnitude_spectrum_log_WOutS = np.log1p(magnitude_spectrum_WOutS)  # log1p to avoid log(0)
    # copute magnitud and aply log scale
    magnitude_spectrum_WS = np.abs(FThologram_WS)
    magnitude_spectrum_log_WS = np.log1p(magnitude_spectrum_WS)  # log1p to avoid log(0)

    #PlotImagen(magnitude_spectrum_log_WOutS, magnitude_spectrum_log_WS, "FT hologram without Speckle", "FT hologram with Speckle")

    # Unwrap the phase of the compensated field and compare with theoretical propagation
    unwrap_teo_WOutS = unwrap_phase(np.angle(objFiltWOutS_NONR))
    unwrap_teo = unwrap_phase(np.angle(objFiltWS_NONR))
    ######################################
    return (
        hologramWOutS_NONR, hologramWS_NONR, np.angle(objFiltWOutS_NONR), np.angle(objFiltWS_NONR) ,unwrap_teo_WOutS, unwrap_teo
            )

def R_Hologram_simulation(name):
    # Load and preprocess the image
    Sample, M, N, m, n = Load_Image(name, 768, 768)
    Invert_Sample = Contrast(Sample)
    
    # Plot the original and contrast-adjusted images
    #Plot1Imagen(Sample, "Sample")
    #Plot1Imagen(Invert_Sample, "Invert_Sample")
    
    # Normalize the inverted image and add speckle noise
    sample, normalize = scale2(Invert_Sample, Phase_factor=False, factor=2*np.pi)
    sample_Speckle = Speckle(normalize)
    _ , sample_Speckle = scale2(sample_Speckle, Phase_factor=False, factor=2*np.pi)
    # Plot the normalized sample and the sample with speckle noise
    #PlotImagen(sample_Speckle1, normalize, "Sample Speckle", "Normalize sample")

    ######## Realistic model parameters ########

    # Setup parameters for the simulation
    lambda_ = 0.633  # Source's wavelength in microns
    k = 2 * np.pi / lambda_  # Wavenumber
    dxy = 3.75  # Pixel size in microns
    mo = 40.0  # Magnification of the microscope objective
    moNA = 0.65  # Numerical aperture of the objective
    tubLens = 200000  # Focal length of the tube lens in microns

    ###################################################

    # Calculate the imaging system parameters
    objetiveFocal = tubLens / mo  # Focal length of the objective
    pitch_firstLens = lambda_ * objetiveFocal / (M * (dxy / mo))  # Pixel pitch for the first lens

    # Calculate the pupil's system parameters
    radius = (moNA * tubLens) / (mo * pitch_firstLens)  # Pupil radius
    resc = round(radius) * 1.5  # Effective radius of the pupil
    pupil = np.zeros((M, N))  # Initialize the pupil function
    distances = np.sqrt(m**2 + n**2)  # Distance matrix from the center
    #print("distance: ", distances.shape)
    pupil[distances <= resc] = 1  # Set the pupil function to 1 within the effective radius
    #print("resc: ", resc)

    pitch_tubeLens = (lambda_ * tubLens) / pitch_firstLens * M  # Pixel pitch for the tube lens

    ###################################################

    # Center of the image
    fx_0 = M / 2
    fy_0 = N / 2
    
    # Generate coordinate grids for the Fourier domain
    m, n, Fx, Fy = coordenadas(M, N)

    # Generate the reference wave
    # x, y = aleatorio(M, N, resc)  # Random coordinates for the reference wave
    x, y = Random_Coordinates(M, N, resc)  # Generate random coordinates for the reference wave
    fx_max = x
    fy_max = y
    
    # Create the reference wave
    ref_wave = Utils.reference_wave(M, N, m, n, lambda_, pitch_firstLens, fx_max, fy_max, k, fx_0, fy_0)
    
    ######################################
    
    # Create the object waves
    objectWOutS = np.exp(1j * normalize)
    objectWS = np.exp(1j * sample_Speckle)
    
    ######################################

    # Compute the Fourier Transforms of the object waves
    FTobjWOutS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectWOutS)))
    FTobjWS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectWS)))

    ######################################

    # Apply the pupil function and perform inverse Fourier Transform
    objFiltWOutS_R = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FTobjWOutS * pupil)))
    objFiltWS_R = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(FTobjWS * pupil)))

    ######################################
  
    # Generate the holograms
    hologramWOutS_R = np.abs(ref_wave + objFiltWOutS_R)**2
    hologramWS_R = np.abs(ref_wave + objFiltWS_R)**2
    #plt.imsave('hologramWS_R.png', hologramWS_R, cmap='gray')
    # Plot the holograms
    # Plot the four holograms

    PlotImagen(hologramWOutS_R, hologramWS_R, "Hologram WOutS Realistic", "Hologram WS Realistic")

    # Compute and plot the Fourier Transforms of the holograms
    fft_holo_WOutS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologramWOutS_R)))
    #fft_holo_WOutS = 20 * np.log(np.abs(fft_holo_WOutS))
    fft_holo_WS = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hologramWS_R)))
    #fft_holo_WS = 20 * np.log(np.abs(fft_holo_WS))

    #Plot1Imagen(fft_holo_WOutS, "FT hologram WOutS")
    Plot1Imagen(fft_holo_WS, "FT hologram WS")

    ######################################

    # Compensate for field distortions
    # Unwrap the phase of the compensated field and compare with theoretical propagation
    unwrap_teo_WOutS = unwrap_phase(np.angle(objFiltWOutS_R))
    #PlotImagen(np.angle(field_compensateWOutS_R), np.angle(objPropagated_WOutS), "Object filter phase rec WOutS R", "Object filter phase R")

    unwrap_teo = unwrap_phase(np.angle(objFiltWS_R))
    #PlotImagen(np.angle(field_compensateWS_R), np.angle(objPropagated_WS), "Object filter phase rec WS R", "Object filter phase R")
    ######################################
    # Return the results
    return (
        hologramWOutS_R, hologramWS_R, np.angle(objFiltWOutS_R), np.angle(objFiltWS_R), unwrap_teo_WOutS, unwrap_teo
    )

def simulate_and_save_holograms(input_folder, output_base_folder):
    # Create output folders
    output_folders = [
        "hologramWOutS_NOTR", "hologramWS_NOTR", "objPropagated_WOutS_NOTR", "objPropagated_WS_NOTR",
        "unwrap_teo_WOutS_NOTR", "unwrap_teo_NOTR"
    ]
    
    for folder in output_folders:
        os.makedirs(os.path.join(output_base_folder, folder), exist_ok=True)
    
    # Process each image in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            image_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            
            # Run the hologram simulation
            results = Not_R_Hologram_simulation(image_path)
            
            # Save the results in corresponding folders
            for result, folder in zip(results, output_folders):
                output_path = os.path.join(output_base_folder, folder, f"{base_name}.png")
                save_image(result, output_path)

input_folder = "Blood_cell"
output_base_folder = "Simulation_NONR"
#simulate_and_save_holograms(input_folder, output_base_folder)
R_Hologram_simulation(r"C:\Users\Usuario\Documents\Emmanuel\Semestre_9\Avanzado_2\Codigos\Blood_cell\15.png")
#Not_R_Hologram_simulation(r"C:\Users\Usuario\Documents\Emmanuel\Semestre_9\Avanzado_2\Codigos\Blood_cell\15.png")

