import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, mapping
import json
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt


class PropertyTaxCalculator:
    standard_area = 144
    standard_area_price = 200
    

    def __init__(self, index, data):
        self.index = index  
        self.data = data
        self.gid = 1
        
    
    def rate_of_area(self, latitude, longitude):
        geojson_file = 'data/berekuso-partition.geojson'
        
        # Read the GeoJSON file into a GeoDataFrame
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
            
        
        point = Point(longitude, latitude)

        for feature in geojson_data['features']:
            polygon = shape(feature['geometry'])
            if polygon.contains(point):
                self.gid = feature['properties']['gid']
                return feature['properties']['rate']
                
        return None
 
        

    def compute_location_tax(self, area_tax):
        # Location factor could be an index or score representing the location value
        # print(latitude, longitude)
        latitude = self.data['latitude'].to_list()[0]
        longitude = self.data['longitude'].to_list()[0]
        
        rate_of_area = self.rate_of_area(latitude, longitude)
        
        if rate_of_area:
            return round(area_tax * rate_of_area, 2)
            
        return area_tax
        
        
    def compute_rooms_tax(self):
        pass
        

    def compute_area_tax(self):
        area_of_building = self.data['area_in_m'].to_list()[0]
        # Area in square meters
        # print(area_of_building)
        return round(area_of_building * PropertyTaxCalculator.standard_area_price / PropertyTaxCalculator.standard_area, 2)
        

        
    def compute_area_foliage_index(self):                    
        
        # Define the paths to the red and NIR band files
        # red_band_path = 'data/satellite/LC09_L2SP_193056_20240528_20240530_02_T1_SR_B4.TIF'
        # nir_band_path = 'data/satellite/LC09_L2SP_193056_20240528_20240530_02_T1_SR_B5.TIF'
        
        # Normalized Difference Vegetation Index (NDVI) approach
        
        # red_band_path = 'data/satellite/cropped4.TIF'
        # nir_band_path = 'data/satellite/cropped5.TIF'
        
        image1 = 'data/satellite/area4.png'
        image2 = 'data/satellite/whole_area.png'
             
       # Open the red and NIR bands as separate rasterio datasets
        with rasterio.open(image1) as image1, rasterio.open(image2) as image2:
            red1 = image1.read(1).astype(float)
            nir1 = image1.read(2).astype(float)
            
            red2 = image2.read(1).astype(float)
            nir2 = image2.read(2).astype(float)
        
        
        # Calculate NDVI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi1 = (nir1 - red1) / (nir1 + red1)
            ndvi1[np.isnan(ndvi1)] = 0  # Set NaNs to 0 for visualization
            
            ndvi2 = (nir2 - red2) / (nir2 + red2)
            ndvi2[np.isnan(ndvi2)] = 0


        # Calculate the mean NDVI
        mean_ndvi1 = np.mean(ndvi1)
        print(f"Mean NDVI1: {mean_ndvi1}")
        
        mean_ndvi2 = np.mean(ndvi2)
        print(f"Mean NDVI2: {mean_ndvi2}")
        
        vegetation_threshold = 0.2
        vegetation_area = np.sum(ndvi1 > vegetation_threshold)
        print(vegetation_area)
        total_area = ndvi1.size 
        vegetation_percentage = (vegetation_area / total_area) * 100
        print(f'Vegetation Area1: {vegetation_percentage}%')
        
        # vegetation_threshold = 0
        vegetation_area = np.sum(ndvi2 > vegetation_threshold)
        print(vegetation_area)
        total_area = ndvi2.size 
        vegetation_percentage = (vegetation_area / total_area) * 100
        print(f'Vegetation Area2: {vegetation_percentage}%')
        
 
        # Plot the NDVI
        plt.figure(figsize=(10, 10))
        plt.imshow(ndvi1, cmap='RdYlGn')
        plt.colorbar(label='NDVI')
        plt.title('NDVI')
        plt.xlabel('Column #')
        plt.ylabel('Row #')
        plt.show()
        
        # Plot the NDVI
        plt.figure(figsize=(10, 10))
        plt.imshow(ndvi2, cmap='RdYlGn')
        plt.colorbar(label='NDVI')
        plt.title('NDVI')
        plt.xlabel('Column #')
        plt.ylabel('Row #')
        plt.show()
        
        return vegetation_percentage
        
        
        
    def compute_building_foliage(self):
        pass
    

    def compute_foliage_tax(self, tax):
    
        # Foliage amount in percentage             
        foliage_in_area = self.compute_area_foliage_index()
        foliage_around_building = self.compute_building_foliage()
        
        # foliage_amount = foliage_in_area/foliage_around_building * tax
        foliage_amount = 10
        
        return foliage_amount
        

    def compute_total_tax(self):
                
        # property tax rate from area of building
        area_tax = self.compute_area_tax()
        
        # property tax rate from location of building
        location_tax = self.compute_location_tax(area_tax)
        
        # property tax rate from foliage surrounding building
        foliage_tax = self.compute_foliage_tax(area_tax)
        
        total_tax = location_tax + area_tax - foliage_tax
        # return total_tax
        
        print('working: ', total_tax)


berekuso_data = pd.read_csv('data/berekuso.csv')
index = 198129

building1 = PropertyTaxCalculator(index, berekuso_data[berekuso_data['index'] == index])
building1.compute_total_tax()

# print(f"The total property tax is: ${total_tax:.2f}")
