import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape, mapping
import json
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
import math


class PropertyTaxCalculator:
    standard_area = 20 #a standard room size
    standard_area_room = 1
    

    def __init__(self, index, data):
        self.index = index  
        self.data = data
        self.gid = 1
 
    def spherical_distance(self, x1, y1, x2, y2):
       # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(x1)
        lon1 = math.radians(y1)
        lat2 = math.radians(x2)
        lon2 = math.radians(y2)
        
        # Spherical law of cosines formula
        distance = math.acos(math.sin(lat1) * math.sin(lat2) + 
                             math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371000
        
        return distance
        

    def compute_location_tax(self):
    
        # Location factor could be an index or score representing the location value
        a_lat = self.data['latitude'].to_list()[0]
        a_lon = self.data['longitude'].to_list()[0]
        
        landmarks = {'ashesi campus':(5.75894936,-0.22051224), 'police station':(5.76222986,-0.22394324), 'queenstar hostel':(5.76498651,-0.21820046), 
        'taxi station':(5.76004795,-0.22499349), 'quest hostel':(5.75572838,-0.22662854), 'agri_impact':(5.75976745,-0.23099225), 
        'berekuso_basic':(5.75404406,-0.23133638), 'maranatha hostel':(5.75088916,-0.22884111), 'spice consult':(5.748484,-0.22963367)}
        
        # Initialize variables to track the shortest distance and corresponding landmark
        shortest_distance = float('inf')
        closest_landmark = None
        
        # Loop through the landmarks and calculate the distance to point 'a'
        for key, (lat, lon) in landmarks.items():
            distance = self.spherical_distance(a_lat, a_lon, lat, lon)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_landmark = key
                
        return f"The closest landmark is '{closest_landmark.upper()}' ({shortest_distance:.2f} meters away)"
        
        
    def compute_rooms_tax(self):
        area = area_of_building = self.data['area_in_m'].to_list()[0]
        
        return f'About {round(area/PropertyTaxCalculator.standard_area)} rooms'
        

    def compute_area_tax(self):
        area_of_building = self.data['area_in_m'].to_list()[0]
        # Area in square meters
        # print(area_of_building)
        return round(area_of_building * PropertyTaxCalculator.standard_area_price / PropertyTaxCalculator.standard_area, 2)
        
    
    def area_index(self):
        latitude = self.data['latitude'].to_list()[0]
        longitude = self.data['longitude'].to_list()[0]
        
        geojson_file = 'data/berekuso-partition.geojson'
        
        # Read the GeoJSON file into a GeoDataFrame
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
            
        
        point = Point(longitude, latitude)

        for feature in geojson_data['features']:
            polygon = shape(feature['geometry'])
            if polygon.contains(point):
                self.gid = str(feature['properties']['gid'])
                print('GID: '+ self.gid)
                return self.gid
                
        return None
        
        
    def compute_area_foliage_index(self, image_name):                    
        
        # Define the paths to the red and NIR band files
        image = 'data/satellite/'+image_name+'.png'
             
       # Open the red and NIR bands as separate rasterio datasets
        with rasterio.open(image) as image:
            red = image.read(1).astype(float)
            nir = image.read(2).astype(float)
        
        
        # Calculate NDVI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0  # Set NaNs to 0 for visualization



        # Calculate the mean NDVI
        mean_ndvi = np.mean(ndvi)
        # print(f"Mean NDVI: {mean_ndvi}")
        
        
        vegetation_threshold = 0.2
        vegetation_area = np.sum(ndvi > vegetation_threshold)
        total_area = ndvi.size 
        vegetation_percentage = (vegetation_area / total_area) * 100
        # print(f'Vegetation Area1: {vegetation_percentage}%')
        
        return vegetation_percentage
    

    def compute_foliage_tax(self):
    
        # Foliage of amount area of building in percentage
        area_name = 'area'+self.area_index()
        foliage_in_area = self.compute_area_foliage_index(area_name)
        
        # Foliage of amount Berekuso in percentage  
        foliage_berekuso = self.compute_area_foliage_index('whole_area')
        
        foliage_amount = foliage_in_area/100 *foliage_berekuso
        foliage_amount = foliage_amount/foliage_berekuso*100
        
        return f"Area of property makes {foliage_amount:.2f}% of Berekuso's foliage"
        
    
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


