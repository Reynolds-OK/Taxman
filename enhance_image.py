import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def enlarge_image(input_path, output_path, target_size=(1500, 1500)):
    """
    Enlarge an image to the target size while maintaining the aspect ratio.
    
    Parameters:
    - input_path: Path to the input image file
    - output_path: Path to the output image file
    - target_size: Target size for the output image (default is (1500, 1500))
    """
    with Image.open(input_path) as img:
        # Resize the image to the target size directly, without maintaining aspect ratio
        new_img = img.resize(target_size, Image.LANCZOS)
        
        # Save the enlarged image
        new_img.save(output_path)
        # print(f"Enlarged Image saved at {output_path}")
    
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Image')
    # plt.imshow(img, cmap='gray')
    
    # plt.subplot(1, 2, 2)
    # plt.title('Enlarged Image')
    # plt.imshow(new_img, cmap='gray')
    
    # plt.show()


def histogram_equalization(image_name, path, ex):

    #Enlarge Image to right fit
    input_path = path+image_name+ex
    output_path = path+image_name+'_en'+ex  # Path to save the output image
    
    enlarge_image(input_path, output_path)

    # Load the image
    image = cv2.imread(output_path)

    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Mask all zeros (if any)
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Equalize the CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill the masked values back in the original CDF
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Use the CDF as a LUT (Look-Up Table) to transform the image
    equalized_image = cdf[image]
    
    # Convert the equalized array back to an image
    equalized_img = Image.fromarray(equalized_image)
    
    brightness_reduction=0.5
    
     # Reduce brightness
    if brightness_reduction < 1.0:
        # Convert PIL image to numpy array
        equalized_array = np.array(equalized_img)
        
        # Reduce brightness by scaling pixel values
        equalized_array = (equalized_array * brightness_reduction).clip(0, 255).astype(np.uint8)
        
        # Convert back to PIL image
        equalized_img = Image.fromarray(equalized_array)
    
    # Save the equalized image
    equalized_img.save('capture/starting1/'+image_name+'_hist'+ex)
    
    # print(f"Equalized image saved at {path+image_name+'_hist'+ex}")
    print(f"Enhanced image saved at {path+image_name+'_en'+ex}")
    
    # return image, equalized_image

# # Display the results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title('Equalized Image')
# plt.imshow(equalized_image, cmap='gray')

# plt.show()

def crop_image_from_center(image_path, distance, output_path):
    """
    Crop an image by a specific distance from the center.

    Parameters:
    image_path (str): Path to the input image.
    distance (int): The distance in pixels to crop from the center.
    output_path (str): Path to save the cropped image.

    Returns:
    None
    """
    # Open the image file
    with Image.open(image_path) as img:
        width, height = img.size
        # Calculate the coordinates of the cropping box
        left = (width / 2) - distance
        top = (height / 2) - distance
        right = (width / 2) + distance
        bottom = (height / 2) + distance
        
        # Ensure the crop box is within image bounds
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, width)
        bottom = min(bottom, height)
        
        # Define the cropping box (left, upper, right, lower)
        crop_box = (left, top, right, bottom)
        
        # Crop the image using the defined box
        cropped_img = img.crop(crop_box)
        
        # Save the cropped image to the specified output path
        cropped_img.save(output_path)
        
        
import os

def delete_image(image_path):
    """
    Delete an image file.

    Parameters:
    image_path (str): Path to the image file to be deleted.

    Returns:
    None
    """
    try:
        os.remove(image_path)
        # print(f"Image {image_path} deleted successfully.")
    except FileNotFoundError:
        print(f"Image {image_path} not found.")
    except PermissionError:
        print(f"Permission denied to delete {image_path}.")
    except Exception as e:
        print(f"Error occurred while deleting the image {image_path}: {e}")



# try 
# name = 'area_0'
# path = 'capture/test/'
# ex='.tif'

# crop_image_from_center('capture/output.png', 325, 'capture/cropped1.png')

# Apply histogram equalization
# image, equalized_image = histogram_equalization(name, path, ex)


