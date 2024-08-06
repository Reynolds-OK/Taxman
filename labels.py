import os
from osgeo import gdal
import pandas as pd
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
from shapely import wkt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from capture_data import Pixel_width


def is_within_geo_rectangle(geo_transform, longitude, latitude):
    
    rect_top_left = (geo_transform[0], geo_transform[3])
    
    pixel_x, pixel_y = 1500,1500
    
    rect_lon1, rect_lat1 = rect_top_left
    
    """Convert pixel coordinates to geographic coordinates."""
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    
    rect_bottom_right = (geo_x, geo_y)
    rect_lon2, rect_lat2 = geo_x, geo_y
    
    # Define the rectangle corners
    rect_top_right = (rect_lon2, rect_lat1)
    rect_bottom_left = (rect_lon1, rect_lat2)
    
    # Create a polygon for the rectangle
    rectangle = Polygon([
        rect_top_left,
        rect_top_right,
        rect_bottom_right,
        rect_bottom_left,
        rect_top_left
    ])
    
    # Create a point object for the point
    point = Point(longitude, latitude)
    
    # Check if the point is within the rectangle
    return rectangle.contains(point)
    
    
def create_georeferenced_polygon_boundary_mask_from_wkt(image_size, polygons_wkt_list, top_left_coords, pixel_size, output_filename):
    """
    Create a mask image with the boundary of multiple polygons in white, georeferenced and saved as a TIFF.

    :param image_size: A tuple (width, height) representing the size of the image.
    :param polygons_wkt_list: A list of polygons in WKT format.
    :param top_left_coords: A tuple (lon, lat) representing the top-left corner coordinates of the image.
    :param pixel_size: The size of each pixel in the units of the coordinates (e.g., degrees).
    :param output_filename: The filename to save the output TIFF.
    """
    width, height = image_size
    lon0, lat0 = top_left_coords

    # Create a new black image
    mask = Image.new('L', image_size, 0)  # 'L' mode for grayscale, 0 for black
    draw = ImageDraw.Draw(mask)
    
    # Convert geographic coordinates to image pixel coordinates
    def geo_to_pixel(lon, lat):
        x = int((lon - lon0) / pixel_size)
        y = int((lat0 - lat) / pixel_size)  # lat0 is top, so subtract lat from lat0
        return x, y

    # Process each polygon
    for polygon_wkt in polygons_wkt_list:
        polygon = wkt.loads(polygon_wkt)
        if not isinstance(polygon, Polygon):
            continue  # Skip if not a valid polygon
        polygon_points = list(polygon.exterior.coords)
        pixel_polygon_points = [geo_to_pixel(lon, lat) for lon, lat in polygon_points]
        draw.polygon(pixel_polygon_points, outline=255, fill=255)
    
    # Save the image as a georeferenced TIFF
    mask_array = np.array(mask)
    
    transform = from_origin(lon0, lat0, pixel_size, pixel_size)
    new_dataset = rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=mask_array.dtype,
        crs='EPSG:4326',  # Assuming WGS84 coordinates
        transform=transform,
    )
    
    new_dataset.write(mask_array, 1)
    new_dataset.close()
  
    
def create_label():
    directory = 'capture/after_processing'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    number_of_imgs = len(files) 
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv('data/berekuso.csv')
    
    # Extract specific columns
    columns_of_interest = ['longitude', 'latitude', 'geometry']
    df_subset = df[columns_of_interest]
    
    
    for i in range(number_of_imgs):
        # Open the georeferenced image
        dataset = gdal.Open(f"capture/after_processing/area_{i}.tif")
        
        # Get the geo-transform matrix
        geo_transform = dataset.GetGeoTransform()
        
        polygons_list = []
        # Loop through each row and process the data
        for index, row in df_subset.iterrows():
            longitude = float(row['longitude'])
            latitude = float(row['latitude'])
            
            contains = is_within_geo_rectangle(geo_transform, longitude, latitude)
            
            if contains:
                polygons_list.append(row['geometry'])

        image_size = (1500, 1500) 
        top_left_coords = geo_transform[0], geo_transform[3]
        pixel_size = Pixel_width  
        
        create_georeferenced_polygon_boundary_mask_from_wkt(image_size, polygons_list, top_left_coords, pixel_size, f'capture/training_labels/area_{i}.tif')
        

create_label()