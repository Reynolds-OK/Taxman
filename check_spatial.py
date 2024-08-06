from osgeo import gdal

# Open the georeferenced image
dataset = gdal.Open("capture/training/area_10_en.tif")

# Get the geo-transform matrix
geo_transform = dataset.GetGeoTransform()

def pixel_to_geo(pixel_x, pixel_y, geo_transform):
    """Convert pixel coordinates to geographic coordinates."""
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    
    print(geo_transform[0], geo_transform[1], geo_transform[2], geo_transform[3], geo_transform[4], geo_transform[5])
    
    return geo_x, geo_y

# Example pixel coordinates
pixel_x, pixel_y = 1500,1500

# Convert to geographic coordinates
geo_x, geo_y = pixel_to_geo(pixel_x, pixel_y, geo_transform)
print(f"Geographic coordinates: ({geo_y}, {geo_x})")

#lat anfd long
# in 5.759722, -0.222222
# out 5.756944, -0.219444