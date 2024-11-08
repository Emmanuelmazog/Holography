############################################################# 
# Title: U-Net Implementation for Hologram Phase Reconstruction                 #
#                                                                               #
# Description: This program implements a U-Net architecture for reconstructing  #
# phase information from digital holograms. The implementation uses a           #
# conditional GAN approach with a U-Net generator and a convolutional           #
# discriminator. The program includes data augmentation, custom loss functions, #
# and adaptive learning rate scheduling.                                        #
#                                                                               #
# Features:                                                                     #
# - Custom DataGenerator for efficient data handling and augmentation           #
# - U-Net architecture for the generator                                        #
# - Conditional GAN implementation                                              #
# - Perceptual loss using VGG16                                                 #
# - Adaptive learning rate scheduling                                           #
# - Multiple training configurations                                            #
#                                                                               #
# Authors: Emmanuel Mazo                                                        #
# Applied Optics Group                                                          #
# EAFIT University                                                              #
#                                                                               #
# Version: 2.0                                                                  #
# Last Updated: 2024                                                            #
#############################################################

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Global configuration parameters
INPUT_SHAPE = (256, 256, 1)  # Input image dimensions
BATCH_SIZE = 44              # Number of samples per batch
EPOCHS = 200                 # Total number of training epochs
rotation_angles = [90, 180, 270]  # Angles for data augmentation
num_images_to_augment = 0    # Number of images to augment

########################################################
# Custom Data Generator Class
########################################################
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, hologram_paths, phase_paths, batch_size, input_shape, is_training=False, num_augment=0, rotation_angles=[90, 180, 270], shuffle=True):
        self.hologram_paths = hologram_paths
        self.phase_paths = phase_paths
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.is_training = is_training
        self.num_augment = num_augment
        self.rotation_angles = rotation_angles
        self.shuffle = shuffle
        self.on_epoch_end()

        # Store the original filenames
        self.hologram_filenames = [os.path.basename(path) for path in hologram_paths]
        

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.hologram_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate batch indexes
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        hologram_paths_batch = [self.hologram_paths[k] for k in indexes]
        phase_paths_batch = [self.phase_paths[k] for k in indexes]

        # Generate data for the batch
        X, Y = self.__data_generation(hologram_paths_batch, phase_paths_batch)

        # Get filenames for this batch
        batch_filenames = [self.hologram_filenames[k] for k in indexes]

        # Apply data augmentation if in training mode
        if self.is_training and self.num_augment > 0:
            X_aug, Y_aug = self.__augment_batch(X, Y)
            
            # Reshape augmented data
            X_aug = X_aug.reshape(X_aug.shape[0], *self.input_shape)
            Y_aug = Y_aug.reshape(Y_aug.shape[0], *self.input_shape)
            
            # Combine original and augmented data
            X = np.concatenate([X, X_aug], axis=0)
            Y = np.concatenate([Y, Y_aug], axis=0)

        return X, Y, batch_filenames

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch if required
        self.indexes = np.arange(len(self.hologram_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, hologram_paths_batch, phase_paths_batch):
        X = np.empty((len(hologram_paths_batch), *self.input_shape))
        Y = np.empty((len(phase_paths_batch), *self.input_shape))
        
        for i, (hologram_path, phase_path) in enumerate(zip(hologram_paths_batch, phase_paths_batch)):
            # Load and preprocess images
            hologram = cv2.imread(hologram_path, cv2.IMREAD_GRAYSCALE)
            phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
            hologram = cv2.resize(hologram, (self.input_shape[1], self.input_shape[0]))
            phase = cv2.resize(phase, (self.input_shape[1], self.input_shape[0]))
            # Scale pixel values to [-1, 1]
            hologram = (hologram / 127.5) - 1.0
            phase = (phase / 127.5) - 1.0
            X[i,] = hologram.reshape(*self.input_shape)
            Y[i,] = phase.reshape(*self.input_shape)
        
        return X, Y

    def __augment_batch(self, X, Y):
        X_aug = []
        Y_aug = []
        for _ in range(self.num_augment):
            idx = np.random.randint(0, len(X))
            x_aug, y_aug = self.generate_random_augmentation(X[idx], Y[idx])
            X_aug.append(x_aug)
            Y_aug.append(y_aug)
        return np.array(X_aug), np.array(Y_aug)

    def generate_random_augmentation(self, hologram_image, phase_image):
        # Apply random rotation and flip augmentations
        angle = random.choice(self.rotation_angles)
        hologram_aug = self.__rotate_image(hologram_image, angle)
        phase_aug = self.__rotate_image(phase_image, angle)

        if random.random() < 0.5:
            hologram_aug = cv2.flip(hologram_aug, 0)
            phase_aug = cv2.flip(phase_aug, 0)
        
        if random.random() < 0.5:
            hologram_aug = cv2.flip(hologram_aug, 1)
            phase_aug = cv2.flip(phase_aug, 1)

        return hologram_aug, phase_aug

    def __rotate_image(self, image, angle):
        # Rotate image based on angle
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image


########################################################
# Model Architecture Definitions
########################################################


# Function to build the U-Net model (generator)
def build_generator(input_shape):

    """
    Build U-Net generator model
    
    Args:
        input_shape (tuple): Input image dimensions
        
    Returns:
        tensorflow.keras.Model: Compiled generator model
    """
    inputs = tf.keras.Input(shape=input_shape)

    def encoder_block(input_tensor, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        return x

    conv1 = encoder_block(inputs, 32)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = encoder_block(pool1, 64)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = encoder_block(pool2, 128)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = encoder_block(pool3, 256)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = encoder_block(pool4, 512)

    def decoder_block(input_tensor, concat_tensor, num_filters):
        x = layers.UpSampling2D(size=(2, 2))(input_tensor)
        x = layers.Conv2D(num_filters, (2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Concatenate(axis=3)([concat_tensor, x])
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Decoder blocks for upsampling and concatenating features
    up1 = decoder_block(conv5, conv4, 256)
    up2 = decoder_block(up1, conv3, 128)
    up3 = decoder_block(up2, conv2, 64)
    up4 = decoder_block(up3, conv1, 32)
    # Final output layer with a single channel and 'tanh' activation
    outputs = layers.Conv2D(1, (1, 1), activation='tanh')(up4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
# Discriminator
def build_discriminator(input_shape):
    """
    Build discriminator model
    
    Args:
        input_shape (tuple): Input image dimensions
        
    Returns:
        tensorflow.keras.Model: Compiled discriminator model
    """
    hologram_input = layers.Input(shape=input_shape)
    phase_input = layers.Input(shape=input_shape)
    
    combined_input = layers.Concatenate()([hologram_input, phase_input])
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=[hologram_input, phase_input], outputs=x)

########################################################
# Loss Functions and Model Components
########################################################

# Perceptual loss
def create_perceptual_loss_model():
    """
    Create VGG16-based model for perceptual loss calculation
    
    Returns:
        tensorflow.keras.Model: Model for perceptual loss computation
    """
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    return models.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

perceptual_model = create_perceptual_loss_model()

def perceptual_loss(y_true, y_pred):
    """
    Calculate perceptual loss using VGG16 features
    
    Args:
        y_true (tensor): Ground truth phase images
        y_pred (tensor): Generated phase images
        
    Returns:
        tensor: Computed perceptual loss
    """
    # Repeat single channel to create 3-channel images for VGG
    y_true_3 = tf.repeat(y_true, 3, axis=-1)
    y_pred_3 = tf.repeat(y_pred, 3, axis=-1)
    return tf.reduce_mean(tf.square(perceptual_model(y_true_3) - perceptual_model(y_pred_3)))

# Custom GAN loss
def custom_gan_loss(y_true, y_pred):
    """
    Custom loss function combining L1 and perceptual losses
    
    Args:
        y_true (tensor): Ground truth phase images
        y_pred (tensor): Generated phase images
        
    Returns:
        tensor: Combined loss value
    """
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    perc_loss = perceptual_loss(y_true, y_pred)
    return l1_loss + 0.1 * perc_loss

########################################################
# Visualization and Monitoring Functions
########################################################

def plot_losses(d_losses, g_losses, val_d_losses, val_g_losses):
    """
    Plot training and validation losses
    
    Args:
        d_losses (list): Discriminator training losses
        g_losses (list): Generator training losses
        val_d_losses (list): Discriminator validation losses
        val_g_losses (list): Generator validation losses
    """
    plt.figure(figsize=(10, 5))

    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation losses

    plt.subplot(1, 2, 2)
    plt.plot(val_d_losses, label='Discriminator')
    plt.plot(val_g_losses, label='Generator')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gan_losses.png')
    plt.show()


########################################################
# Learning Rate Adaptation
########################################################

class AdaptiveLearningRateCallback(Callback):
    """
    Custom callback for adaptive learning rate adjustment during training
    """
    def __init__(self, generator, discriminator, patience=5, factor=0.5, min_lr=1e-6):
        """
        Initialize the adaptive learning rate callback
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            patience (int): Number of epochs to wait before reducing learning rate
            factor (float): Factor by which to reduce learning rate
            min_lr (float): Minimum learning rate
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to adjust learning rates if needed
        """
        current = logs.get('val_loss')
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                gen_lr = float(tf.keras.backend.get_value(self.generator.optimizer.lr))
                disc_lr = float(tf.keras.backend.get_value(self.discriminator.optimizer.lr))
                
                new_gen_lr = max(gen_lr * self.factor, self.min_lr)
                new_disc_lr = max(disc_lr * self.factor, self.min_lr)
                
                tf.keras.backend.set_value(self.generator.optimizer.lr, new_gen_lr)
                tf.keras.backend.set_value(self.discriminator.optimizer.lr, new_disc_lr)
                
                print(f'\nEpoch {epoch}: reducing learning rate.')
                print(f'Generator LR: {new_gen_lr:.6f}, Discriminator LR: {new_disc_lr:.6f}')
                
                self.wait = 0

########################################################
# Training Functions
########################################################

# Training function
def train_gan(generator, discriminator, gan, epochs, train_generator, val_generator, lr_callback):
    """
    Main training loop for the GAN
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        gan: Combined GAN model
        epochs (int): Number of training epochs
        train_generator: Training data generator
        val_generator: Validation data generator
        lr_callback: Learning rate callback
        
    Returns:
        tuple: Lists of training and validation losses
    """
    d_losses = []
    g_losses = []
    val_d_losses = []
    val_g_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_d_losses = []
        epoch_g_losses = []
        
        for batch in range(len(train_generator)):
            real_holograms, real_phases, _ = train_generator[batch]
            batch_size = real_holograms.shape[0]

            # Generate fake phases
            fake_phases = generator.predict(real_holograms)

            # Label smoothing
            real_labels = np.ones((batch_size, 1)) * 0.9  # Instead of 1
            fake_labels = np.zeros((batch_size, 1)) + 0.1  # Instead of 0

            # Train discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch([real_holograms, real_phases], real_labels)
            d_loss_fake = discriminator.train_on_batch([real_holograms, fake_phases], fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # Train generator (via GAN model)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(real_holograms, [real_labels, real_phases])

            epoch_d_losses.append(d_loss)
            epoch_g_losses.append(g_loss[0])  # Use the first element of g_loss (assuming it's the total loss)

        # Calculate average losses for the epoch
        d_losses.append(np.mean(epoch_d_losses))
        g_losses.append(np.mean(epoch_g_losses))

        # Validation
        val_holograms, val_phases, _ = next(iter(val_generator))
        val_fake_phases = generator.predict(val_holograms)
        val_real_labels = np.ones((val_holograms.shape[0], 1)) * 0.9
        val_fake_labels = np.zeros((val_holograms.shape[0], 1)) + 0.1
        
        discriminator.trainable = True
        val_d_loss_real = discriminator.test_on_batch([val_holograms, val_phases], val_real_labels)
        val_d_loss_fake = discriminator.test_on_batch([val_holograms, val_fake_phases], val_fake_labels)
        val_d_loss = 0.5 * (val_d_loss_real + val_d_loss_fake)
        
        discriminator.trainable = False
        val_g_loss = gan.test_on_batch(val_holograms, [val_real_labels, val_phases])

        val_d_losses.append(val_d_loss)
        val_g_losses.append(val_g_loss[0])  # Use the first element of val_g_loss (assuming it's the total loss)

        print(f"D loss: {d_losses[-1]}, G loss: {g_losses[-1]}")
        print(f"Val D loss: {val_d_losses[-1]}, Val G loss: {val_g_losses[-1]}")

        # Update learning rates
        lr_callback.on_epoch_end(epoch, logs={'val_loss': val_g_losses[-1]})

    return d_losses, g_losses, val_d_losses, val_g_losses

########################################################
# Image Generation and Saving Functions
########################################################

# Function to save generated images
def save_generated_images(generator, test_generator, save_dir='generated_images'):

    """
    Save generated phase images from the test set
    
    Args:
        generator: Trained generator model
        test_generator: Data generator for test set
        save_dir (str): Directory to save generated images
    """

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save images for each batch in test set
    for i in range(len(test_generator)):
        # Get batch of test data
        holo, phase, batch_filenames = test_generator[i]
        # Generate phase images
        predicted_phase = generator.predict(holo)
        
        # Save each generated image with its original filename
        for j, filename in enumerate(batch_filenames):
            # Convert the predicted phase back to 8-bit format (0-255)
            save_image = (predicted_phase[j, ..., 0] * 127.5 + 127.5).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, filename), save_image)

########################################################
# Main Training Configuration Function
########################################################

def train_cgan_with_configuration(config_name, hologram_folder,
                                phase_folder, output_hologram_folder=None,
                                output_phase_folder=None, 
                                rbcs_ratio=1.0):
    """
    Train the conditional GAN with specified configuration
    
    Args:
        config_name (str): Name of the configuration for saving results
        hologram_folder (str): Path to folder containing hologram images
        phase_folder (str): Path to folder containing phase images
        output_hologram_folder (str, optional): Path to additional hologram images
        output_phase_folder (str, optional): Path to additional phase images
        rbcs_ratio (float): Ratio of RBCs images to use in training
    """
    
    print(f"Training cGAN with configuration: {config_name}")
    
    
    # Load and prepare data
    if output_hologram_folder and output_phase_folder:
        # Load paths for both RBCs and output images
        rbcs_holos = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]
        rbcs_phases = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]
        output_holos = [os.path.join(output_hologram_folder, file) for file in os.listdir(output_hologram_folder)]
        output_phases = [os.path.join(output_phase_folder, file) for file in os.listdir(output_phase_folder)]
        
        # Calculate split sizes
        total_size = len(rbcs_holos)
        rbcs_size = int(total_size * rbcs_ratio)
        output_size = total_size - rbcs_size
        
        # Randomly select from RBCs and Output
        selected_rbcs_holos = random.sample(rbcs_holos, rbcs_size)
        selected_rbcs_phases = [p.replace("Holograms", "Invert_unwrapped_phases") for p in selected_rbcs_holos]
        selected_output_holos = random.sample(output_holos, output_size)
        selected_output_phases = [p.replace("hologramWS_R_256", "unwrap_teo_WOutS_256") for p in selected_output_holos]
        
        # Combine and shuffle
        hologram_paths = selected_rbcs_holos + selected_output_holos
        phase_paths = selected_rbcs_phases + selected_output_phases
    else:
        hologram_paths = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]
        phase_paths = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]
    
    # Combine hologram and phase paths into a list of tuples
    combined = list(zip(hologram_paths, phase_paths))
    # Shuffle the combined list to ensure randomness
    random.shuffle(combined)
    # Unzip the combined list back into separate hologram and phase paths
    hologram_paths, phase_paths = zip(*combined)

    # Split the data into training, validation, and test sets
    holos_train, holos_temp, phases_train, phases_temp = train_test_split(hologram_paths, phase_paths, test_size=0.3, random_state=42)
    holos_val, holos_test, phases_val, phases_test = train_test_split(holos_temp, phases_temp, test_size=0.33, random_state=42)

    # Save the test set file paths for later use
    test_set_paths = {
        'holos_test': holos_test,
        'phases_test': phases_test
    }
    with open(f'{config_name}_test_set_paths.pkl', 'wb') as f:
        pickle.dump(test_set_paths, f)

    # Create data generators for training, validation, and testing
    train_generator = DataGenerator(
        holos_train, phases_train, 
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        is_training=True,
        num_augment=num_images_to_augment // (len(holos_train) // BATCH_SIZE),
        rotation_angles=rotation_angles
    )

    val_generator = DataGenerator(
        holos_val, phases_val, 
        batch_size=BATCH_SIZE, 
        input_shape=INPUT_SHAPE, 
        is_training=False
    )

    test_generator = DataGenerator(
        holos_test, phases_test,
        batch_size=BATCH_SIZE, 
        input_shape=INPUT_SHAPE, 
        is_training=False
    )

    # Build and compile the generator and discriminator models
    generator = build_generator(INPUT_SHAPE)
    discriminator = build_discriminator(INPUT_SHAPE)

    # Define learning rate schedules for both generator and discriminator
    gen_lr_schedule = ExponentialDecay(0.0002, decay_steps=1000, decay_rate=0.96, staircase=True)
    disc_lr_schedule = ExponentialDecay(0.0001, decay_steps=1000, decay_rate=0.96, staircase=True)

    # Create optimizers for both models
    gen_optimizer = Adam(learning_rate=gen_lr_schedule, beta_1=0.5, beta_2=0.999)
    disc_optimizer = Adam(learning_rate=disc_lr_schedule, beta_1=0.5, beta_2=0.999)

    # Compile the generator and discriminator models
    generator.compile(optimizer=gen_optimizer)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy')

    # Freeze the discriminator during GAN training
    discriminator.trainable = False

    # Create the GAN model by connecting the generator and discriminator
    gan_input = layers.Input(shape=INPUT_SHAPE)
    generated_phase = generator(gan_input)
    discriminator_output = discriminator([gan_input, generated_phase])
    gan = models.Model(inputs=gan_input, outputs=[discriminator_output, generated_phase])

    # Compile the GAN model with custom loss functions
    gan.compile(optimizer=gen_optimizer,
                loss=['binary_crossentropy', custom_gan_loss],
                loss_weights=[1, 100])

    # Create a callback for adaptive learning rate adjustment
    lr_callback = AdaptiveLearningRateCallback(generator, discriminator)

    # Train the conditional GAN and record losses
    d_losses, g_losses, val_d_losses, val_g_losses = train_gan(generator, discriminator, gan, EPOCHS, train_generator, val_generator, lr_callback)

    # Plot and save the training losses
    plot_losses(d_losses, g_losses, val_d_losses, val_g_losses)
    plt.savefig(f'{config_name}_losses.png')
    plt.close()

    # Save the trained generator and discriminator models
    generator.save(f'{config_name}_generator_V2.h5')
    discriminator.save(f'{config_name}_discriminator_V2.h5')

    # Evaluate the trained model on the test set
    test_holograms, test_phases, _ = next(iter(test_generator))
    test_generated_phases = generator.predict(test_holograms)

    # Save the generated images from the test set
    save_generated_images(generator, test_generator, save_dir=f'{config_name}_test_predictions')

    # Calculate and print evaluation metrics for the generated phases
    mse = np.mean((test_phases - test_generated_phases) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    print(f"Configuration: {config_name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Peak Signal-to-Noise Ratio: {psnr} dB")

    # Save the evaluation metrics to a text file
    with open(f'{config_name}_metrics.txt', 'w') as f:
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Peak Signal-to-Noise Ratio: {psnr} dB\n")
if __name__ == "__main__":
    main_folder = "/home/user/RBCs_DB_TD"
    output_folder = "/home/user/RBCs_simulated"

    # Configuration 1: RBCs dataset only
    hologram_folder = os.path.join(main_folder, "Holograms")
    phase_folder = os.path.join(main_folder, "Invert_unwrapped_phases")
    train_cgan_with_configuration("RBCs_only", hologram_folder, phase_folder)

    # Configuration 2: 70% RBCs, 30% Output
    output_hologram_folder = os.path.join(output_folder, "hologramWS_R_256")
    output_phase_folder = os.path.join(output_folder, "unwrap_teo_WOutS_256")
    train_cgan_with_configuration("70RBCs_30Output", hologram_folder, phase_folder, output_hologram_folder, output_phase_folder, rbcs_ratio=0.7)

    # Configuration 3: 50% RBCs, 50% Output
    train_cgan_with_configuration("50RBCs_50Output", hologram_folder, phase_folder, output_hologram_folder, output_phase_folder, rbcs_ratio=0.5)

    # Configuration 4: 100% Output
    train_cgan_with_configuration("100Output", output_hologram_folder, output_phase_folder)

print("All configurations have been trained and evaluated.")

