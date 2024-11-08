#############################################################
# Title: U-Net Implementation for Hologram Phase Reconstruction                               
#                                                                              
# Description: This program implements a U-Net architecture for reconstructing
# phase information from digital holograms                               
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(43)
random.seed(42)

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

        # Load the images in grayscale and in 2D
        hologram = tf.keras.preprocessing.image.load_img(hologram_path, 
                                                       target_size=input_shape, 
                                                       color_mode='grayscale')
        phase = tf.keras.preprocessing.image.load_img(phase_path, 
                                                    target_size=input_shape, 
                                                    color_mode='grayscale')

        # Convert images to arrays and normalize to [0,1]
        hologram_array = tf.keras.preprocessing.image.img_to_array(hologram)[:, :, 0] / 255.0
        phase_array = tf.keras.preprocessing.image.img_to_array(phase)[:, :, 0] / 255.0

        X.append(hologram_array)
        Y.append(phase_array)

    return np.array(X), np.array(Y), image_names

def build_unet(input_shape):
    """
    Build a U-Net model for image-to-image translation.
    
    Architecture:
    - Encoder: 4 blocks of double convolution with max pooling
    - Bottleneck: Double convolution
    - Decoder: 4 blocks of upsampling, concatenation, and double convolution
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder block definition
    def encoder_block(input_tensor, num_filters):
        """Create an encoder block with double convolution and batch normalization"""
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    # Encoder pathway with skip connections
    conv1 = encoder_block(inputs, 32)          # First encoding block
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = encoder_block(pool1, 64)           # Second encoding block
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = encoder_block(pool2, 128)          # Third encoding block
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = encoder_block(pool3, 256)          # Fourth encoding block
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = encoder_block(pool4, 512)          # Bottleneck layer

    # Decoder block definition
    def decoder_block(input_tensor, concat_tensor, num_filters):
        """Create a decoder block with upsampling, concatenation, and double convolution"""
        x = layers.UpSampling2D(size=(2, 2))(input_tensor)
        x = layers.Conv2D(num_filters, (2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)             # Dropout for regularization
        x = layers.Concatenate(axis=3)([concat_tensor, x])  # Skip connection
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Decoder pathway with skip connections
    up1 = decoder_block(conv5, conv4, 256)     # First decoding block
    up2 = decoder_block(up1, conv3, 128)       # Second decoding block
    up3 = decoder_block(up2, conv2, 64)        # Third decoding block
    up4 = decoder_block(up3, conv1, 32)        # Fourth decoding block

    # Output layer with tanh activation for phase reconstruction
    outputs = layers.Conv2D(1, (1, 1), activation='tanh')(up4)

    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model

# Model and training parameters
input_shape = (256, 256, 1)
batch_size = 44
epochs = 500
hologram_folder = "holo_WS_prueba"
phase_folder = "obj_phase_WS_prueba"

# Load and preprocess dataset
holos, phases, image_names = load_data(hologram_folder, phase_folder, input_shape)

# Split dataset into training, validation, and test sets
holos_train, holos_temp, phases_train, phases_temp, image_names_train, image_names_temp = train_test_split(
    holos, phases, image_names, test_size=0.3, random_state=42)
holos_val, holos_test, phases_val, phases_test, image_names_val, image_names_test = train_test_split(
    holos_temp, phases_temp, image_names_temp, test_size=0.33, random_state=42)

# Create and compile the U-Net model
unet_model = build_unet(input_shape)

# Define training callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    verbose=1,
    restore_best_weights=True
)
checkpoint_callback = ModelCheckpoint(
    filepath="checkpoint/model_Unet_paper_{epoch:02d}.h5",
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = unet_model.fit(
    holos_train, phases_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(holos_val, phases_val),
    callbacks=[early_stopping, checkpoint_callback]
)

# Save the trained model
trained_model_path = "modelo_unet_500E_44B_10k_Paper.h5"
unet_model.save(trained_model_path)

# Generate and save reconstructions for test set
indices = np.random.choice(range(len(holos_test)), 100, replace=False)
test_images = holos_test[indices]
test_image_names = [image_names_test[i] for i in indices]

# Create output directory if it doesn't exist
if not os.path.exists("reconstruccion_modelo_U_net_paper"):
    os.makedirs("reconstruccion_modelo_U_net_paper")

# Generate and save reconstructions
for i, image in enumerate(test_images):
    reconstructed_image = unet_model.predict(np.expand_dims(image, axis=0))[0]
    cv2.imwrite(os.path.join("reconstruccion_modelo_U_net_paper", 
                            f"{test_image_names[i]}.png"), 
                reconstructed_image * 255.0)

# Plot and save training history
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_plot_U_net_paper.png")
plt.show()