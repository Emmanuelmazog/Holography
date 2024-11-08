#############################################################
# Title: Image generator algorithm                               
#                                                                              
# Description: This program is a specialized tool for generating 
# synthetic phase images and their wrapped versions                               
#                                                                              
# Authors: Emmanuel Mazo Gómez, Carlos Trujillo                                                     
# Applied optics group
# EAFIT university                   
#                                           
# Medellín, Colombia.                                                            
#                                                                              
# Email: emazog@eafit.edu.co
# Version: 2.0 (2024)                                                           
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint, choice

def Plot1Imagen(original, title, cmap='gray'):
    """
    Display a single image with a colorbar showing intensity values.
    
    Args:
        original: The image array to display
        title: Title for the plot
        cmap: Colormap to use (default: 'gray')
    """
    plt.imshow(original, cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Intensidad')
    plt.show()
    
def operate_terms(x, y, cross_term, c, function_list, fuction_name, x_name, y_name, operation_name, x_terms=True, y_terms=True, cross_terms=True, multiply=True):
    """
    Generate image terms by applying mathematical functions to input coordinates and combining them.
    
    Args:
        x, y: Input coordinate arrays
        cross_term: Result of operation between x and y (e.g., x+y, x-y, x*y)
        c: Counter for selecting functions from function_list
        function_list: List of mathematical functions to apply (e.g., sin, cos)
        fuction_name: List of function names for generating the image name
        x_name, y_name: Names of x and y terms for generating the image name
        operation_name: Name of the operation applied between x and y
        x_terms, y_terms, cross_terms: Booleans to control which terms to include
        multiply: If True, multiply terms; if False, add them
    
    Returns:
        tuple: (Combined image array, updated counter, generated image name)
    """
    if multiply:
        mod = 1  # Multiplicative identity
    else:
        mod = 0  # Additive identity

    # Process x terms if enabled
    if x_terms:
        img1 = (function_list[c % len(function_list)](x))**2
        nameimg1 = f"({fuction_name[c % len(function_list)]}({x_name}))A2"
        c += 1
    else:
        img1 = mod
        nameimg1 = ""

    # Process y terms if enabled
    if y_terms:
        img2 = (function_list[c % len(function_list)](y))**2
        nameimg2 = f"({fuction_name[c % len(function_list)]}({y_name}))A2"
        c += 1
    else:
        img2 = mod
        nameimg2 = ""

    # Process cross terms if enabled
    if cross_terms:
        img3 = (function_list[c % len(function_list)](cross_term))**2
        nameimg3 = f"({fuction_name[c % len(function_list)]}({operation_name}))A2"
        c += 1
    else:
        img3 = mod
        nameimg3 = ""

    # Combine terms either through multiplication or addition
    if multiply:
        img = img1 * img2 * img3
    else:
        img = img1 + img2 + img3
    
    image_name = nameimg1 + nameimg2 + nameimg3
    return img, c, image_name

def generate_image(x, y, nmax, function_list, fuction_name, x_key, y_key, multiply=True, **kwargs):
    """
    Generate a complete image by combining multiple mathematical operations and functions.
    
    Args:
        x, y: Input coordinate arrays
        nmax: Maximum value for normalization
        function_list: List of mathematical functions to apply
        fuction_name: List of function names
        x_key, y_key: Keys identifying the type of x and y terms
        multiply: If True, multiply terms; if False, add them
        **kwargs: Additional arguments passed to operate_terms
    
    Returns:
        tuple: (Generated image array, image name describing the operations used)
    """
    # Define possible operations between x and y
    operations = {
        "x+y": x + y,
        "x-y": x - y,
        "xPy": x * y
    }
    operation_key = choice(list(operations.keys()))
    operation_value = operations[operation_key]

    # Generate initial image terms
    c = 0
    img, c, image_name = operate_terms(x, y, operation_value, c, function_list, fuction_name, 
                                     x_key, y_key, operation_key, **kwargs, multiply=multiply)

    # Generate and combine additional terms
    while c < len(function_list):
        new_img, c, new_image_name = operate_terms(x, y, operation_value, c, function_list, 
                                                 fuction_name, x_key, y_key, operation_key, 
                                                 **kwargs, multiply=multiply)
        if multiply:
            img *= new_img
            image_name = image_name + "P" + new_image_name
        else:
            img += new_img
            image_name = image_name + "+" + new_image_name

    # Normalize the image
    img = img / abs(img).max() * nmax
    return img, image_name

# Main execution block
if __name__ == '__main__':
    # Setup directories for storing generated images
    data_folder = 'data'
    real_folder = 'real_simple'
    wrapped_folder = 'wrapped_simple'
    n_images = 10000  # Number of images to generate
    start_number = 1
    img_size = 256   # Image dimensions
    size = complex(f'{img_size}j')

    # Create necessary directories if they don't exist
    if data_folder not in os.listdir('.'):
        os.mkdir(data_folder)
    if real_folder not in os.listdir(data_folder):
        os.mkdir(os.path.join(data_folder, real_folder))
    if wrapped_folder not in os.listdir(data_folder):
        os.mkdir(os.path.join(data_folder, wrapped_folder))

    # Create coordinate grid
    x, y = np.mgrid[-1:1:size, -1:1:size]
    nmax = 2 * np.pi  # Maximum phase value

    # Define available mathematical functions
    function_list = {
        "sin": np.sin,
        "cos": np.cos
    }
    
    # Randomly select 2-3 functions for image generation
    fuction_list = []
    fuction_name = []
    for i in range(randint(2, 3)):
        fuction_key = choice(list(function_list.keys()))
        fuction_list.append(function_list[fuction_key])
        fuction_name.append(fuction_key)

    # Define possible variations of x and y terms
    x_variations = {
        "xA3": x ** 3,
        "xA2": x ** 2,
        "x": x
    }
    y_variations = {
        "yA3": y ** 3,
        "yA2": y ** 2,
        "y": y
    }

    # Generate multiple images with random parameters
    for n in range(n_images):
        # Generate random factors for x and y terms
        x_factor = randint(-3, 3)
        if x_factor == 0:
            x_factor = 1
        y_factor = randint(-3, 3)
        if y_factor == 0:
            y_factor = 1

        # Set generation parameters
        multiply = True
        x_terms = bool(randint(0, 1))
        y_terms = bool(randint(0, 1))
        cross_terms = True

        # Ensure at least one term is enabled
        while not x_terms and not y_terms and not cross_terms:
            x_terms = bool(randint(0, 1))
            y_terms = bool(randint(0, 1))
            cross_terms = bool(randint(0, 1))

        # Randomly select x and y variations
        x_key = choice(list(x_variations.keys()))
        x_value = x_variations[x_key]
        y_key = choice(list(y_variations.keys()))
        y_value = y_variations[y_key]

        # Generate the image
        img, image_name = generate_image(x_value * x_factor, y_value * y_factor, nmax, 
                                       fuction_list, fuction_name, x_key, y_key, 
                                       multiply=multiply, x_terms=x_terms, 
                                       y_terms=y_terms, cross_terms=cross_terms)

        # Save the original (unwrapped) image
        filenamefuction = f'{n + start_number}.png'
        plt.imsave(os.path.join(data_folder, real_folder, filenamefuction), img, cmap='gray')
        plt.close()

        # Generate and save the wrapped version of the image
        wrapped_img = np.angle(np.exp(1j * img))
        plt.imsave(os.path.join(data_folder, wrapped_folder, filenamefuction), wrapped_img, cmap='gray')
        plt.close()