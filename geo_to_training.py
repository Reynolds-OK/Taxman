from add_geospatial import georeference_image
from enhance_image import delete_image


for i in range(38):
    # Input and output image paths
    
    input_image_path = f'capture/training/area_{i}_en.png'
    output_image_path = f'capture/training/area_{i}_en.tif'
    
    
    # Georeference the image
    georeference_image(i, input_image_path, output_image_path)
    delete_image(input_image_path)