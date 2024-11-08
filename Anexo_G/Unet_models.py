############################################################# 
# Title: U-Net Implementation for Hologram Phase Reconstruction                 #
#                                                                               #
# Description: This program implements a U-Net architecture for reconstructing  #
# phase information from digital holograms. The implementation uses a           #
# traditional U-Net approach with data augmentation and custom training         #
# configurations.                                                               #
#                                                                               #
# Features:                                                                     #
# - Custom DataGenerator for efficient data handling and augmentation           #
# - U-Net architecture with skip connections                                    #
# - Multiple training configurations for different dataset combinations         #
# - Comprehensive evaluation and visualization tools                            #
#                                                                               #
# Authors: Emmanuel Mazo                                                        #
# Applied Optics Group                                                          #
# EAFIT University                                                              #
#                                                                               #
# Version: 1.0                                                                  #
# Last Updated: 2024                                                            #
#############################################################

# Import necessary libraries for the U-Net model and data processing
import os  # For file and directory operations
import numpy as np  # For numerical operations
import tensorflow as tf  # For building and training the neural network
import matplotlib.pyplot as plt  # For plotting loss graphs
from tensorflow.keras import layers, models  # For building the model layers
import cv2  # For image processing
import random  # For random operations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from tensorflow.keras.optimizers import Adam  # For the Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # For model training callbacks
from tensorflow.keras.initializers import RandomNormal  # For weight initialization
import pickle  # For saving and loading Python objects

# Set a seed for reproducibility to ensure consistent results across runs
np.random.seed(43)  # Seed for NumPy random number generator
random.seed(42)  # Seed for Python's random number generator
tf.random.set_seed(43)  # Seed for TensorFlow random number generator

# DataGenerator class to handle data loading and augmentation
class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator class for efficient batch loading and augmentation of hologram data.
    
    Features:
    - Batch-wise data loading
    - Real-time data augmentation
    - Memory-efficient processing
    - Support for training and validation modes
    
    Methods:
    - __init__: Initialize generator with paths and parameters
    - __len__: Calculate number of batches per epoch
    - __getitem__: Generate one batch of data
    - on_epoch_end: Shuffle data between epochs
    - __data_generation: Load and preprocess images
    - __augment_batch: Generate augmented data
    - generate_random_augmentation: Apply random augmentations
    - __rotate_image: Rotate image by specified angle
    """
    def __init__(self, hologram_paths, phase_paths, batch_size, input_shape, is_training=False, num_augment=0, rotation_angles=[90, 180, 270], shuffle=True):
        # Initialize the data generator with paths, batch size, input shape, and augmentation parameters
        self.hologram_paths = hologram_paths  # List of hologram image paths
        self.phase_paths = phase_paths  # List of phase image paths
        self.batch_size = batch_size  # Number of samples per batch
        self.input_shape = input_shape  # Shape of the input images
        self.is_training = is_training  # Flag to indicate if the generator is for training
        self.num_augment = num_augment  # Number of augmentations to apply
        self.rotation_angles = rotation_angles  # List of angles for rotation augmentations
        self.shuffle = shuffle  # Flag to indicate if data should be shuffled
        self.on_epoch_end()  # Shuffle data at the start

        # Store the original filenames for saving generated images later
        self.hologram_filenames = [os.path.basename(path) for path in hologram_paths]

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(len(self.hologram_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # Get the indexes for the current batch
        hologram_paths_batch = [self.hologram_paths[k] for k in indexes]  # Get hologram paths for the batch
        phase_paths_batch = [self.phase_paths[k] for k in indexes]  # Get phase paths for the batch

        # Generate data for the current batch
        X, Y = self.__data_generation(hologram_paths_batch, phase_paths_batch)

        # If training and augmentations are specified, augment the data
        if self.is_training and self.num_augment > 0:
            X_aug, Y_aug = self.__augment_batch(X, Y)  # Generate augmented data
            X_aug = X_aug.reshape(X_aug.shape[0], *self.input_shape)  # Reshape augmented data
            Y_aug = Y_aug.reshape(Y_aug.shape[0], *self.input_shape)  # Reshape augmented labels
            X = np.concatenate([X, X_aug], axis=0)  # Concatenate original and augmented data
            Y = np.concatenate([Y, Y_aug], axis=0)  # Concatenate original and augmented labels

        return X, Y  # Return the batch of data and labels

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch if required
        self.indexes = np.arange(len(self.hologram_paths))  # Create an array of indexes
        if self.shuffle:
            np.random.shuffle(self.indexes)  # Shuffle the indexes

    def __data_generation(self, hologram_paths_batch, phase_paths_batch):
        # Generate data for the given batch of hologram and phase paths
        X = np.empty((len(hologram_paths_batch), *self.input_shape))  # Initialize empty array for holograms
        Y = np.empty((len(phase_paths_batch), *self.input_shape))  # Initialize empty array for phases
        
        for i, (hologram_path, phase_path) in enumerate(zip(hologram_paths_batch, phase_paths_batch)):
            # Load and preprocess images
            hologram = cv2.imread(hologram_path, cv2.IMREAD_GRAYSCALE)  # Read hologram image
            phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)  # Read phase image
            hologram = cv2.resize(hologram, (self.input_shape[1], self.input_shape[0]))  # Resize hologram
            phase = cv2.resize(phase, (self.input_shape[1], self.input_shape[0]))  # Resize phase
            # Scale pixel values to [-1, 1]
            hologram = (hologram / 127.5) - 1.0  # Normalize hologram
            phase = (phase / 127.5) - 1.0  # Normalize phase
            X[i,] = hologram.reshape(*self.input_shape)  # Reshape and assign hologram to batch
            Y[i,] = phase.reshape(*self.input_shape)  # Reshape and assign phase to batch
        
        return X, Y  # Return the generated data and labels

    def __augment_batch(self, X, Y):
        # Generate augmented data for the batch
        X_aug = []  # List to hold augmented holograms
        Y_aug = []  # List to hold augmented phases
        for _ in range(self.num_augment):
            idx = np.random.randint(0, len(X))  # Randomly select an index for augmentation
            x_aug, y_aug = self.generate_random_augmentation(X[idx], Y[idx])  # Generate random augmentation
            X_aug.append(x_aug)  # Append augmented hologram
            Y_aug.append(y_aug)  # Append augmented phase
        return np.array(X_aug), np.array(Y_aug)  # Return augmented data and labels

    def generate_random_augmentation(self, hologram_image, phase_image):
        # Apply random rotation and flip augmentations
        angle = random.choice(self.rotation_angles)  # Randomly select a rotation angle
        hologram_aug = self.__rotate_image(hologram_image, angle)  # Rotate hologram
        phase_aug = self.__rotate_image(phase_image, angle)  # Rotate phase

        # Randomly flip the images vertically
        if random.random() < 0.5:
            hologram_aug = cv2.flip(hologram_aug, 0)  # Flip hologram vertically
            phase_aug = cv2.flip(phase_aug, 0)  # Flip phase vertically
        
        # Randomly flip the images horizontally
        if random.random() < 0.5:
            hologram_aug = cv2.flip(hologram_aug, 1)  # Flip hologram horizontally
            phase_aug = cv2.flip(phase_aug, 1)  # Flip phase horizontally

        return hologram_aug, phase_aug  # Return augmented images

    def __rotate_image(self, image, angle):
        # Rotate image based on angle
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)  # Rotate 180 degrees
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 90 degrees counterclockwise
        return image  # Return the original image if no rotation is applied

def build_unet(input_shape, kernel_init):
    """
    Constructs a U-Net model for hologram phase reconstruction.
    
    Features:
    - Encoder-decoder architecture with skip connections
    - Batch normalization for training stability
    - Dropout layers for regularization
    - LeakyReLU activation in encoder
    - ReLU activation in decoder
    
    Args:
        input_shape: Dimensions of input images
        kernel_init: Weight initialization strategy
    
    Returns:
        Compiled U-Net model
    """

    # Build U-Net model architecture
    inputs = tf.keras.Input(shape=input_shape)  # Define input layer
    
    # Encoder block function to create layers
    def encoder_block(input_tensor, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(input_tensor)  # Convolutional layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.LeakyReLU(alpha=0.2)(x)  # Leaky ReLU activation
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(x)  # Second convolutional layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.LeakyReLU(alpha=0.2)(x)  # Leaky ReLU activation
        return x  # Return the output of the encoder block
    
    # Create encoder layers
    conv1 = encoder_block(inputs, 32)  # First encoder block
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # Max pooling

    conv2 = encoder_block(pool1, 64)  # Second encoder block
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # Max pooling

    conv3 = encoder_block(pool2, 128)  # Third encoder block
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # Max pooling

    conv4 = encoder_block(pool3, 256)  # Fourth encoder block
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)  # Max pooling

    # Bottleneck layer
    conv5 = encoder_block(pool4, 512)  # Bottleneck encoder block

    # Decoder block function to create layers
    def decoder_block(input_tensor, concat_tensor, num_filters):
        x = layers.UpSampling2D(size=(2, 2))(input_tensor)  # Upsampling
        x = layers.Conv2D(num_filters, (2, 2), padding='same', kernel_initializer=kernel_init)(x)  # Convolutional layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.ReLU()(x)  # ReLU activation
        x = layers.Dropout(0.5)(x)  # Dropout for regularization
        x = layers.Concatenate(axis=3)([concat_tensor, x])  # Concatenate with corresponding encoder output
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(x)  # Convolutional layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.ReLU()(x)  # ReLU activation
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(x)  # Convolutional layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.ReLU()(x)  # ReLU activation
        return x  # Return the output of the decoder block

    # Create decoder layers
    conv6 = decoder_block(conv5, conv4, 256)  # First decoder block
    conv7 = decoder_block(conv6, conv3, 128)  # Second decoder block
    conv8 = decoder_block(conv7, conv2, 64)  # Third decoder block
    conv9 = decoder_block(conv8, conv1, 32)  # Fourth decoder block

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='tanh')(conv9)  # Final convolutional layer

    model = models.Model(inputs, outputs)  # Create the model
    return model  # Return the model

def save_generated_images(generator, test_generator, save_dir='generated_images'):
    """
    Saves model predictions on test data.
    
    Features:
    - Creates output directory if needed
    - Maintains original filename associations
    - Converts predictions to appropriate pixel range
    - Saves images in standard format
    
    Args:
        generator: Trained model
        test_generator: Data generator for test set
        save_dir: Output directory path
    """
    # Create a directory to save generated images
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    for i in range(len(test_generator)):
        holo, phase = test_generator[i]  # Get hologram and phase from the test generator
        predicted_phase = generator.predict(holo)  # Generate predictions
        
        start_idx = i * test_generator.batch_size  # Calculate start index for filenames
        end_idx = start_idx + holo.shape[0]  # Calculate end index for filenames
        batch_filenames = test_generator.hologram_filenames[start_idx:end_idx]  # Get filenames for the current batch
        
        for j, filename in enumerate(batch_filenames):
            # Save the predicted phase images
            cv2.imwrite(os.path.join(save_dir, filename), (predicted_phase[j, ..., 0] * 127.5 + 127.5).astype(np.uint8))

# Function to train and evaluate a model
def train_and_evaluate_model(train_generator, val_generator, test_generator, model, epochs, batch_size, model_name):
    """
    Trains and evaluates the U-Net model with comprehensive monitoring.
    
    Features:
    - Early stopping to prevent overfitting
    - Model checkpointing for best weights
    - Adaptive learning rate
    - Loss visualization
    - Model evaluation on test set
    
    Args:
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator
        model: U-Net model
        epochs: Number of training epochs
        batch_size: Batch size
        model_name: Name for saving model and results
    
    Returns:
        Training history and test loss
    """
    # Compile the model with optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.999), loss='mean_squared_error')

    # Callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)  # Early stopping
    checkpoint_callback = ModelCheckpoint(
        filepath=f'{model_name}.h5',  # Filepath to save the model
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=False  # Change this to False to save the full model
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)  # Reduce learning rate

    # Train the model
    history = model.fit(
        train_generator,  # Training data
        validation_data=val_generator,  # Validation data
        epochs=epochs,  # Number of epochs
        callbacks=[early_stopping, checkpoint_callback, reduce_lr]  # Callbacks
    )

    # Evaluate the model on the test set
    test_loss = model.evaluate(test_generator)  # Evaluate test loss

    # Plot and save loss
    plot_and_save_loss(history, f'{model_name}_loss_plot.png')  # Plot loss graph

    # Save the final model (optional, as the best model is already saved by checkpoint_callback)
    model.save(f'{model_name}_final.h5')  # Save the final model

    # Save generated images from the model
    save_generated_images(model, test_generator, save_dir=f'{model_name}_test_predictions_U_Net')

    return history, test_loss  # Return training history and test loss

# Function to plot and save loss
def plot_and_save_loss(history, save_path):
     
    '''Visualizes and saves training progress.
    
    Features:
    - Plots training and validation loss
    - Includes grid for better readability
    - Saves plot to specified path
    - Proper axis labeling and legend
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    '''
    
    # Create a plot for training and validation loss
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(history.history['loss'], label='Training Loss')  # Plot training loss
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
    plt.title('Model Loss')  # Set title
    plt.ylabel('Loss')  # Set y-axis label
    plt.xlabel('Epoch')  # Set x-axis label
    plt.legend(loc='upper right')  # Set legend location
    plt.grid(True)  # Enable grid
    plt.savefig(save_path)  # Save the plot
    plt.close()  # Close the plot

# Main execution function
def main():
    """
    Main execution function implementing multiple training configurations.
    
    Configurations:
    1. RBCs dataset only
    2. 70% RBCs, 30% Output
    3. 50% RBCs, 50% Output
    4. 100% Output
    
    Features:
    - Data loading and preprocessing
    - Dataset splitting and combination
    - Model training and evaluation
    - Results saving and documentation
    
    Each configuration includes:
    - Custom data splitting
    - Model training
    - Performance evaluation
    - Results documentation
    """
    # Directories for data
    main_folder = "/home/user/RBCs_DB_TD"  # Main folder containing datasets
    output_folder = "/home/user/RBCs_simulated"  # Output folder for results
    
    # Fixed hyperparameters for training
    batch_size = 44  # Batch size for training
    num_augment = 0  # Number of augmentations to apply
    
    # Configuration 1: RBCs dataset only
    hologram_folder = os.path.join(main_folder, "Holograms")  # Folder containing hologram images
    phase_folder = os.path.join(main_folder, "Invert_unwrapped_phases")  # Folder containing phase images
    
    model_name = f"unet_RBCs"  # Model name for saving

    # Load and split data into training, validation, and test sets
    hologram_paths = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]  # Get hologram paths
    phase_paths = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]  # Get phase paths
    
    # Split data into training, validation, and test sets
    holos_train, holos_temp, phases_train, phases_temp = train_test_split(hologram_paths, phase_paths, test_size=0.3, random_state=42)
    holos_val, holos_test, phases_val, phases_test = train_test_split(holos_temp, phases_temp, test_size=0.33, random_state=42)
    
    # Save the test set file paths for later use
    test_set_paths = {
        'holos_test': holos_test,
        'phases_test': phases_test
    }
    with open(f'{model_name}_test_set_paths.pkl', 'wb') as f:
        pickle.dump(test_set_paths, f)  # Save test set paths to a pickle file

    # Create data generators for training, validation, and testing
    train_generator = DataGenerator(holos_train, phases_train, batch_size=batch_size, input_shape=(256,256,1), is_training=True, num_augment=num_augment)
    val_generator = DataGenerator(holos_val, phases_val, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    test_generator = DataGenerator(holos_test, phases_test, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    
    # Build and train the U-Net model
    model = build_unet((256,256,1), RandomNormal(stddev=0.02))  # Build the model
    history, test_loss = train_and_evaluate_model(train_generator, val_generator, test_generator, model, epochs=250, batch_size=batch_size, model_name=model_name)  # Train the model
    
    # Save results of the training
    with open(f"{model_name}_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")  # Write test loss to file
        f.write(f"Best Validation Loss: {min(history.history['val_loss'])}\n")  # Write best validation loss to file

    # Configuration 2: 70% RBCs, 30% Output
    output_hologram_folder = os.path.join(output_folder, "hologramWS_R_256")  # Output hologram folder
    output_phase_folder = os.path.join(output_folder, "unwrap_teo_WOutS_256")  # Output phase folder
    
    model_name = f"unet_70RBCs_30Output"  # Model name for saving
    
    # Load data from both datasets
    rbcs_holos = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]  # RBCs hologram paths
    rbcs_phases = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]  # RBCs phase paths
    output_holos = [os.path.join(output_hologram_folder, file) for file in os.listdir(output_hologram_folder)]  # Output hologram paths
    output_phases = [os.path.join(output_phase_folder, file) for file in os.listdir(output_phase_folder)]  # Output phase paths

    # Verify that both datasets have the same number of elements
    assert len(rbcs_holos) == len(output_holos), "RBCs and Output datasets should have the same number of elements"

    # Calculate the total number of samples (which is the same as the length of either dataset)
    total_samples = len(rbcs_holos)  # or len(output_holos), they should be the same

    # Calculate the number of samples for each split
    rbcs_size = int(0.7 * total_samples)  # Number of RBCs samples
    output_size = total_samples - rbcs_size  # Number of Output samples

    print(f"Total samples: {total_samples}")  # Print total samples
    print(f"RBCs samples: {rbcs_size} ({rbcs_size/total_samples:.2%})")  # Print RBCs samples
    print(f"Output samples: {output_size} ({output_size/total_samples:.2%})")  # Print Output samples

    # Randomly select from RBCs and Output
    selected_rbcs_holos = random.sample(rbcs_holos, rbcs_size)  # Randomly select RBCs holograms
    selected_rbcs_phases = [p.replace("Holograms", "Invert_unwrapped_phases") for p in selected_rbcs_holos]  # Get corresponding phases
    selected_output_holos = random.sample(output_holos, output_size)  # Randomly select Output holograms
    selected_output_phases = [p.replace("hologramWS_R_256", "unwrap_teo_WOutS_256") for p in selected_output_holos]  # Get corresponding phases

    # Combine and shuffle the selected datasets
    combined_holos = selected_rbcs_holos + selected_output_holos  # Combine holograms
    combined_phases = selected_rbcs_phases + selected_output_phases  # Combine phases
    combined = list(zip(combined_holos, combined_phases))  # Zip combined holograms and phases
    random.shuffle(combined)  # Shuffle the combined dataset
    combined_holos, combined_phases = zip(*combined)  # Unzip the combined dataset
    
    # Split into train, val, test
    holos_train, holos_temp, phases_train, phases_temp = train_test_split(combined_holos, combined_phases, test_size=0.3, random_state=42)  # Split into training and temp
    holos_val, holos_test, phases_val, phases_test = train_test_split(holos_temp, phases_temp, test_size=0.33, random_state=42)  # Split temp into validation and test
    # Save the test set file paths
    test_set_paths = {
        'holos_test': holos_test,
        'phases_test': phases_test
    }
    with open(f'{model_name}_test_set_paths.pkl', 'wb') as f:
        pickle.dump(test_set_paths, f)  # Save test set paths to a pickle file
    # Create data generators for training, validation, and testing
    train_generator = DataGenerator(holos_train, phases_train, batch_size=batch_size, input_shape=(256,256,1), is_training=True, num_augment=num_augment)
    val_generator = DataGenerator(holos_val, phases_val, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    test_generator = DataGenerator(holos_test, phases_test, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    
    # Build and train the U-Net model
    model = build_unet((256,256,1), RandomNormal(stddev=0.02))  # Build the model
    history, test_loss = train_and_evaluate_model(train_generator, val_generator, test_generator, model, epochs=250, batch_size=batch_size, model_name=model_name)  # Train the model

    # Save results of the training
    with open(f"{model_name}_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")  # Write test loss to file
        f.write(f"Best Validation Loss: {min(history.history['val_loss'])}\n")  # Write best validation loss to file

    # Configuration 3: 50% RBCs, 50% Output
    model_name = f"unet_50RBCs_50Output"  # Model name for saving
    
    # Load and split data
    rbcs_holos = [os.path.join(hologram_folder, file) for file in os.listdir(hologram_folder)]  # RBCs hologram paths
    rbcs_phases = [os.path.join(phase_folder, file) for file in os.listdir(phase_folder)]  # RBCs phase paths
    output_holos = [os.path.join(output_hologram_folder, file) for file in os.listdir(output_hologram_folder)]  # Output hologram paths
    output_phases = [os.path.join(output_phase_folder, file) for file in os.listdir(output_phase_folder)]  # Output phase paths
    
    # Calculate split sizes
    total_size = min(len(rbcs_holos), len(output_holos))  # Get the minimum size of both datasets
    rbcs_size = output_size = total_size // 2  # Split evenly between RBCs and Output
    
    # Randomly select from RBCs and Output
    selected_rbcs_holos = random.sample(rbcs_holos, rbcs_size)  # Randomly select RBCs holograms
    selected_rbcs_phases = [p.replace("Holograms", "Invert_unwrapped_phases") for p in selected_rbcs_holos]  # Get corresponding phases
    selected_output_holos = random.sample(output_holos, output_size)  # Randomly select Output holograms
    selected_output_phases = [p.replace("hologramWS_R_256", "unwrap_teo_WOutS_256") for p in selected_output_holos]  # Get corresponding phases
    
    # Combine and shuffle the selected datasets
    combined_holos = selected_rbcs_holos + selected_output_holos  # Combine holograms
    combined_phases = selected_rbcs_phases + selected_output_phases  # Combine phases
    combined = list(zip(combined_holos, combined_phases))  # Zip combined holograms and phases
    random.shuffle(combined)  # Shuffle the combined dataset
    combined_holos, combined_phases = zip(*combined)  # Unzip the combined dataset
    
    # Split into train, val, test
    holos_train, holos_temp, phases_train, phases_temp = train_test_split(combined_holos, combined_phases, test_size=0.3, random_state=42)  # Split into training and temp
    holos_val, holos_test, phases_val, phases_test = train_test_split(holos_temp, phases_temp, test_size=0.33, random_state=42)  # Split temp into validation and test
    # Save the test set file paths
    test_set_paths = {
        'holos_test': holos_test,
        'phases_test': phases_test
    }
    with open(f'{model_name}_test_set_paths.pkl', 'wb') as f:
        pickle.dump(test_set_paths, f)  # Save test set paths to a pickle file
    # Create data generators for training, validation, and testing
    train_generator = DataGenerator(holos_train, phases_train, batch_size=batch_size, input_shape=(256,256,1), is_training=True, num_augment=num_augment)
    val_generator = DataGenerator(holos_val, phases_val, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    test_generator = DataGenerator(holos_test, phases_test, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    
    # Build and train the U-Net model
    model = build_unet((256,256,1), RandomNormal(stddev=0.02))  # Build the model
    history, test_loss = train_and_evaluate_model(train_generator, val_generator, test_generator, model, epochs=250, batch_size=batch_size, model_name=model_name)  # Train the model
    
    # Save results of the training
    with open(f"{model_name}_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")  # Write test loss to file
        f.write(f"Best Validation Loss: {min(history.history['val_loss'])}\n")  # Write best validation loss to file

    # Configuration 4: 100% Output
    model_name = f"unet_100Output"  # Model name for saving
    
    # Load and split data
    output_holos = [os.path.join(output_hologram_folder, file) for file in os.listdir(output_hologram_folder)]  # Output hologram paths
    output_phases = [os.path.join(output_phase_folder, file) for file in os.listdir(output_phase_folder)]  # Output phase paths
    
    # Split into train, val, test
    holos_train, holos_temp, phases_train, phases_temp = train_test_split(output_holos, output_phases, test_size=0.3, random_state=42)  # Split into training and temp
    holos_val, holos_test, phases_val, phases_test = train_test_split(holos_temp, phases_temp, test_size=0.33, random_state=42)  # Split temp into validation and test
    # Save the test set file paths
    test_set_paths = {
        'holos_test': holos_test,
        'phases_test': phases_test
    }
    with open(f'{model_name}_test_set_paths.pkl', 'wb') as f:
        pickle.dump(test_set_paths, f)  # Save test set paths to a pickle file
    # Create data generators for training, validation, and testing
    train_generator = DataGenerator(holos_train, phases_train, batch_size=batch_size, input_shape=(256,256,1), is_training=True, num_augment=num_augment)
    val_generator = DataGenerator(holos_val, phases_val, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    test_generator = DataGenerator(holos_test, phases_test, batch_size=batch_size, input_shape=(256,256,1), is_training=False)
    
    # Build and train the U-Net model
    model = build_unet((256,256,1), RandomNormal(stddev=0.02))  # Build the model
    history, test_loss = train_and_evaluate_model(train_generator, val_generator, test_generator, model, epochs=250, batch_size=batch_size, model_name=model_name)  # Train the model
    
    # Save results of the training
    with open(f"{model_name}_results.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")  # Write test loss to file
        f.write(f"Best Validation Loss: {min(history.history['val_loss'])}\n")  # Write best validation loss to file

if __name__ == "__main__":
    """
    Entry point of the program.
    Executes the main function when the script is run directly.
    """
    main()  # Execute the main function