import rasterio

def check_geospatial_data(tiff_path):
    with rasterio.open(tiff_path) as src:
        if src.crs and src.transform:
            print(f"File {tiff_path} contains geospatial data.")
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
        else:
            print(f"File {tiff_path} does not contain geospatial data.")

# Example usage
tiff_path = 'capture/after_processing/area_0.tif'
check_geospatial_data(tiff_path)