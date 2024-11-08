#############################################################
# Title: Model Evaluation for Hologram Phase Reconstruction                               
#                                                                              
# Description: This program evaluates and compares the performance of 
# Autoencoder and U-Net models using SSIM and MSE metrics with enhanced
# visualization                              
#                                                                              
# Authors: Emmanuel Mazo
# Applied optics group
# EAFIT university                   
#############################################################

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(43)

def load_data(hologram_folder, phase_folder, input_shape):
    """
    Load and preprocess hologram and phase images from specified folders.
    
    Args:
        hologram_folder: Path to folder containing hologram images
        phase_folder: Path to folder containing phase images
        input_shape: Desired dimensions for input images
        
    Returns:
        tuple: (hologram_arrays, phase_arrays, image_names)
    """
    hologram_paths = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]
    phase_paths = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]
    X = []
    Y = []
    image_names = []  # Lista para almacenar los nombres de las imágenes originales
    
    for hologram_path, phase_path in zip(hologram_paths, phase_paths):
        # Get original image name
        image_name = os.path.splitext(os.path.basename(hologram_path))[0]
        image_names.append(image_name)

        # Load grayscale images
        hologram = tf.keras.preprocessing.image.load_img(hologram_path, 
                                                       target_size=input_shape, 
                                                       color_mode='grayscale')
        phase = tf.keras.preprocessing.image.load_img(phase_path, 
                                                    target_size=input_shape, 
                                                    color_mode='grayscale')

        # Convert to normalized arrays
        hologram_array = tf.keras.preprocessing.image.img_to_array(hologram)[:, :, 0] / 255.0
        phase_array = tf.keras.preprocessing.image.img_to_array(phase)[:, :, 0] / 255.0

        X.append(hologram_array)
        Y.append(phase_array)

    return np.array(X), np.array(Y), image_names

def randomimg(indicador=True):
    """
    Select test images for evaluation.
    
    Args:
        indicador: If True, select 100 random images; if False, use all test images
        
    Returns:
        tuple: (test_holograms, test_phases, test_image_names)
    """
    if indicador:
        # Select 100 random test images
        indices = np.random.choice(range(len(holos_test)), 100, replace=False)
        test_holos = holos_test[indices]
        test_phases = phases_test[indices]
        test_image_names = [image_names_test[i] for i in indices]
    else:
        # Use all test images
        test_holos = holos_test
        test_phases = phases_test
        test_image_names = image_names_test
        
    return test_holos, test_phases, test_image_names

def generate():
    """
    Generate and save reconstructed images using the autoencoder model.
    """
    # Create output directory if needed
    if not os.path.exists("reconstruccion_autoencoder"):
        os.makedirs("reconstruccion_autoencoder")

    # Generate reconstructions for each test image
    for i, (image, image_name) in enumerate(zip(test_holos, test_image_names)):
        reconstructed_image = autoencoder.predict(np.expand_dims(image, axis=0))[0]
        cv2.imwrite(os.path.join("reconstruccion_autoencoder", 
                                f"{image_name}.png"), 
                   reconstructed_image * 255.0)

def metric():
    """
    Calculate and visualize comparison metrics between Autoencoder and U-Net models.
    
    Metrics:
    - SSIM (Structural Similarity Index)
    - MSE (Mean Squared Error)
    
    Creates enhanced visualizations with:
    - Publication-quality histograms
    - Grid lines
    - Scientific notation
    - Customized fonts and sizes
    """
    # Initialize metric arrays
    ssim_scores_autoencoder = []
    ssim_scores_unet = []
    mse_autoencoder = []
    mse_unet = []

    # Calculate metrics for each test image
    for i, (image, image_name) in enumerate(zip(test_holos, test_image_names)):
        # Generate predictions from both models
        reconstructed_image_autoencoder = autoencoder.predict(np.expand_dims(image, axis=0))[0]
        reconstructed_image_Unet = Unet.predict(np.expand_dims(image, axis=0))[0]

        # Reshape predictions to 2D
        transformed_autoencoder = np.squeeze(reconstructed_image_autoencoder)
        transformed_unet = np.squeeze(reconstructed_image_Unet)

        # Calculate SSIM scores
        ssim_index_autoencoder = ssim(test_phases[i], transformed_autoencoder, 
                                    data_range=transformed_autoencoder.max() - transformed_autoencoder.min())
        ssim_scores_autoencoder.append(ssim_index_autoencoder)

        ssim_index_unet = ssim(test_phases[i], transformed_unet, 
                              data_range=transformed_unet.max() - transformed_unet.min())
        ssim_scores_unet.append(ssim_index_unet)

        # Calculate MSE scores
        mse_index_autoencoder = mean_squared_error(test_phases[i], transformed_autoencoder)
        mse_autoencoder.append(mse_index_autoencoder)

        mse_index_unet = mean_squared_error(test_phases[i], transformed_unet)
        mse_unet.append(mse_index_unet)

    # Calculate average metrics
    mean_ssim_autoencoder = np.mean(ssim_scores_autoencoder)
    mean_ssim_unet = np.mean(ssim_scores_unet)
    mean_mse_autoencoder = np.mean(mse_autoencoder)
    mean_mse_unet = np.mean(mse_unet)

    # Print average metrics
    print(f"Average SSIM Autoencoder: {mean_ssim_autoencoder}")
    print(f"Average SSIM Unet: {mean_ssim_unet}")
    print(f"Average MSE Autoencoder: {mean_mse_autoencoder}")
    print(f"Average MSE Unet: {mean_mse_unet}")

    # Create enhanced SSIM histogram
    plt.figure(figsize=(7, 6))
    plt.hist(ssim_scores_unet, bins=20, alpha=0.5, label='U-Net', color='b')
    plt.hist(ssim_scores_autoencoder, bins=20, alpha=0.5, label='AutoEncoder', color='r')
    plt.xlabel('SSIM', fontsize=18)
    plt.ylabel('Frecuencia', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Histograma de Puntajes SSIM', fontsize=20)
    plt.legend(loc='center left', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save SSIM histogram
    plt.savefig('histogram_ssim.svg', format='svg')
    plt.savefig('histogram_ssim.png', dpi=300)
    plt.show()

    # Create enhanced MSE histogram
    plt.figure(figsize=(7, 6))
    plt.hist(mse_unet, bins=20, alpha=0.5, label='U-Net', color='b', edgecolor='black')
    plt.hist(mse_autoencoder, bins=20, alpha=0.5, label='AutoEncoder', color='r', edgecolor='black')
    plt.xlabel('MSE', fontsize=18)
    plt.ylabel('Frecuencia', fontsize=18)
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Histograma de Puntajes MSE', fontsize=20)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save MSE histogram
    plt.savefig('histogram_mse.svg', format='svg')
    plt.savefig('histogram_mse.png', dpi=300)
    plt.show()

# Model and data parameters
input_shape = (256, 256, 1)
hologram_folder = "holo_WOutS"
phase_folder = "obj_phase_WOutS"

# Load and preprocess the dataset
holos, phases, image_names = load_data(hologram_folder, phase_folder, input_shape)

# Split dataset into train, validation and test sets
holos_train, holos_temp, phases_train, phases_temp, image_names_train, image_names_temp = train_test_split(
    holos, phases, image_names, test_size=0.3, random_state=42)
holos_val, holos_test, phases_val, phases_test, image_names_val, image_names_test = train_test_split(
    holos_temp, phases_temp, image_names_temp, test_size=0.33, random_state=42)

# Load pre-trained models
autoencoder_model = "modelo_autoencoder_500E_44B_10k_paper.keras"
Unet_model = "Unet_paper_resume.keras"
autoencoder = load_model(autoencoder_model)
Unet = load_model(Unet_model)

# Select test images
test_holos, test_phases, test_image_names = randomimg(indicador=False)

# Run evaluation
metric()
#generate()