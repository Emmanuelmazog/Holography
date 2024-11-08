#############################################################
# Title: Autoencoder for Hologram Phase Reconstruction                               
#                                                                              
# Description: This program implements a deep learning approach to reconstruct
# phase information from digital holograms using an autoencoder architecture                               
#                                                                              
# Authors: Emmanuel Mazo
# Email: emazog@eafit.edu.co
# Applied optics group
# EAFIT university  
# Version 1.0 (2024)                 
#############################################################

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(43)
random.seed(43)

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
    image_names = []  # List to store original image names
    
    for hologram_path, phase_path in zip(hologram_paths, phase_paths):
        # Get the original image name
        image_name = os.path.splitext(os.path.basename(hologram_path))[0]
        image_names.append(image_name)

        # Load images in grayscale
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

def build_autoencoder(input_shape):
    """
    Build an autoencoder model for phase reconstruction.
    
    Architecture:
    - Encoder: Series of convolutional and pooling layers
    - Bottleneck: Dense encoding of features
    - Decoder: Series of transposed convolutions for upsampling
    
    Args:
        input_shape: Shape of input images (height, width, channels)
    
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder block definition
    def encoder_block(input_tensor, num_filters):
        """Create an encoder block with Conv2D, BatchNorm, LeakyReLU, and MaxPooling"""
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    # Encoder pathway
    conv1 = encoder_block(inputs, 32)     # First encoding layer
    conv2 = encoder_block(conv1, 64)      # Second encoding layer
    conv3 = encoder_block(conv2, 128)     # Third encoding layer
    conv4 = encoder_block(conv3, 256)     # Fourth encoding layer
    
    # Bottleneck
    conv5 = encoder_block(conv4, 512)     # Bottleneck layer

    # Decoder block definition
    def decoder_block(input_tensor, num_filters):
        """Create a decoder block with Conv2DTranspose, BatchNorm, and ReLU"""
        x = layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Decoder pathway
    up1 = decoder_block(conv5, 256)       # First decoding layer
    up2 = decoder_block(up1, 128)         # Second decoding layer
    up3 = decoder_block(up2, 64)          # Third decoding layer
    up4 = decoder_block(up3, 32)          # Fourth decoding layer
    up5 = layers.UpSampling2D((2, 2))(up4)  # Final upsampling

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(up5)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary() 
    return model

# Model and training configuration parameters
input_shape = (256, 256, 1)     # Input image dimensions
batch_size = 44                 # Batch size for training
epochs = 500                    # Maximum number of training epochs
hologram_folder = "holo_WOutS"  # Input holograms folder
phase_folder = "obj_phase_WOutS"  # Target phase images folder

# Load and preprocess the dataset
holos, phases, image_names = load_data(hologram_folder, phase_folder, input_shape)

# Split dataset into training, validation, and test sets (70%, 20%, 10%)
holos_train, holos_temp, phases_train, phases_temp, image_names_train, image_names_temp = train_test_split(
    holos, phases, image_names, test_size=0.3, random_state=42)
holos_val, holos_test, phases_val, phases_test, image_names_val, image_names_test = train_test_split(
    holos_temp, phases_temp, image_names_temp, test_size=0.33, random_state=42)

# Create and compile the autoencoder model
autoencoder_model = build_autoencoder(input_shape)

# Define training callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1,
    restore_best_weights=True
)
checkpoint_callback = ModelCheckpoint(
    filepath="checkpoint/model_Autoencoder_paper_{epoch:02d}.keras",
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = autoencoder_model.fit(
    holos_train, phases_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(holos_val, phases_val),
    callbacks=[early_stopping, checkpoint_callback]
)

# Save the trained model
trained_model_path = "modelo_autoencoder_500E_44B_10k_Paper.keras"
autoencoder_model.save(trained_model_path)

# Generate and save reconstructions for test set
indices = np.random.choice(range(len(holos_test)), 100, replace=False)
test_images = holos_test[indices]
test_image_names = [image_names_test[i] for i in indices]

# Create output directory if it doesn't exist
if not os.path.exists("reconstruccion_modelo_autoencoder_paper"):
    os.makedirs("reconstruccion_modelo_autoencoder_paper")

# Generate and save reconstructions
for i, image in enumerate(test_images):
    reconstructed_image = autoencoder_model.predict(np.expand_dims(image, axis=0))[0]
    cv2.imwrite(os.path.join("reconstruccion_modelo_autoencoder_paper", 
                            f"{test_image_names[i]}.png"), 
                reconstructed_image * 255.0)

# Plot and save training history
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_plot_autoencoder_paper.png")
plt.show()