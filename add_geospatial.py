from osgeo import gdal, osr
from coor_parse import dms_to_decimal, parse_dms
from capture_data import Edges, Pixel_height, Pixel_width



def georeference_image(index, input_image_path, output_image_path):
    """
    Georeferences an image using given GCPs and EPSG code.
    
    Parameters:
        input_image_path (str): Path to the input image file.
        output_image_path (str): Path to save the georeferenced image.
        gcps (list of gdal.GCP): List of Ground Control Points.
        epsg_code (int): EPSG code of the spatial reference system.
    """
    
    # transform coordinates to longitude and latitude
    lat_deg, lat_min, lat_sec, lat_dir = parse_dms(Edges[index][0])
    lon_deg, lon_min, lon_sec, lon_dir = parse_dms(Edges[index][1])
    
    latitude = dms_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
    longitude = dms_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
    
    # These values need to be adjusted based on your specific georeferencing needs
    # top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height
    top_left_x = longitude  # longitude of the top-left corner
    top_left_y = latitude   # latitude of the top-left corner
    pixel_width = Pixel_width  # pixel size in longitude
    pixel_height = Pixel_height  # pixel size in latitude (negative because the origin is at the top-left corner)
    
    # EPSG code for WGS 84
    epsg_code = 4326

    # Open the input image
    src_ds = gdal.Open(input_image_path)
    
    
    if src_ds is None:
        raise Exception(f"Could not open {input_image_path}")
        
        
    # Get image dimensions
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize
    
    geotransform = (top_left_x, pixel_width, 0, top_left_y, 0, pixel_height)
    
    #Create a new dataset with the same dimensions and number of bands as the source image
    driver = gdal.GetDriverByName('GTiff')
    
    dst_ds = driver.Create(output_image_path, x_size, y_size, src_ds.RasterCount, gdal.GDT_Byte)
    
    # Set the geotransform
    dst_ds.SetGeoTransform(geotransform)
    
    # Set the projection (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)  # EPSG code for WGS84
    dst_ds.SetProjection(srs.ExportToWkt())
    
    # Copy the raster data from the source PNG to the new dataset
    for i in range(1, src_ds.RasterCount + 1):
        band = src_ds.GetRasterBand(i)
        data = band.ReadAsArray()
        dst_ds.GetRasterBand(i).WriteArray(data)
    
    # Close the datasets
    src_ds = None
    dst_ds = None
    
    print(f"Georeferenced image saved as {output_image_path}")



# Input and output image paths
# name = 'pred_mask_1'
# index = 0
# input_image_path = f'capture/area_{0}_en.png'
# output_image_path = f'capture/area_{0}_en.tif'


# # Georeference the image
# georeference_image(index, input_image_path, output_image_path)

