import numpy as np
import os
import cv2
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.geometry import MultiPolygon
import rasterio
from rasterio.features import rasterize
from PIL import Image
from osgeo import gdal
from test_coor import calculate_area
from shapely.ops import unary_union
from shapely.affinity import scale
from shapely import wkt

def _perpendicular_distance(point, start, end):
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    :param point: The point to measure distance from
    :param start: The start point of the line segment
    :param end: The end point of the line segment
    :return: The perpendicular distance from the point to the line segment
    """
    line_mag = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    if line_mag < 1e-10:
        return np.sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2)
    
    u = ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1])) / line_mag**2
    u = max(0, min(1, u))
    closest_point = (start[0] + u * (end[0] - start[0]), start[1] + u * (end[1] - start[1]))
    
    return np.sqrt((point[0] - closest_point[0])**2 + (point[1] - closest_point[1])**2)
    

def _douglas_peucker(coords, tolerance):
    """
    Recursively apply the Douglas-Peucker algorithm to simplify a curve.
    
    :param coords: List of coordinates (Nx2 numpy array)
    :param tolerance: The tolerance parameter to control the level of simplification
    :return: A simplified list of coordinates
    """
    # Base case: if the list of coordinates is very small, return the original coordinates
    if len(coords) < 3:
        return coords

    # Find the point with the maximum distance from the line segment
    start = coords[0]
    end = coords[-1]
    max_dist = 0
    index = 0
    for i in range(1, len(coords) - 1):
        dist = _perpendicular_distance(coords[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    # If the maximum distance is greater than the tolerance, recursively simplify
    if max_dist > tolerance:
        left = _douglas_peucker(coords[:index+1], tolerance)
        right = _douglas_peucker(coords[index:], tolerance)
        return np.concatenate([left[:-1], right])
    else:
        return np.array([start, end])

#simplifies the polygon by removing points that do not significantly alter its shape.
def douglas_peucker(polygon, tolerance):
    """
    Simplify the given polygon using the Douglas-Peucker algorithm.
    
    :param polygon: A shapely.geometry.Polygon object
    :param tolerance: The tolerance parameter to control the level of simplification
    :return: A simplified shapely.geometry.Polygon object
    """
    if polygon.exterior is None:
        raise ValueError("Polygon must have an exterior ring.")
    
    coords = list(polygon.exterior.coords)
    
    if len(coords) < 4:
        # Not enough poin
        return polygon
        
    # Convert Polygon to a list of coordinates
    coords = np.array(coords)
    
    # Apply Douglas-Peucker algorithm to simplify the polygon
    simplified_coords = _douglas_peucker(coords, tolerance)
    
    
    # Create a new Polygon with simplified coordinates
    # print(simplified_coords)
    simplified_polygon = Polygon(simplified_coords)
    
    return simplified_polygon
    

def polygons_to_tiff(polygons):
    """
    Convert a list of shapely polygons to a black and white TIFF image.
    
    Parameters:
    - polygons: List of shapely.geometry.Polygon
    - out_path: Path to the output TIFF file
    - width: Width of the output image
    - height: Height of the output image
    - transform: Affine transformation for the output image
    """
    out_path = 'maskk.png'
    width, height = 1500, 1500
    transform = rasterio.transform.from_origin(0, 0, 1, 1) 
    
    # Create a blank (black) image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Rasterize the polygons into the image
    rasterize(
        [(polygon, 1) for polygon in polygons],
        out_shape=image.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image * 255) 

    # Save the PIL image as a PNG file
    pil_image.save(out_path)

    print(f"PNG image saved at {out_path}")
    

def pixel_to_geo(mask_path, pixel_x, pixel_y):

    # Open the georeferenced image
    dataset = gdal.Open(mask_path)

    # Get the geo-transform matrix
    geo_transform = dataset.GetGeoTransform()

    """Convert pixel coordinates to geographic coordinates."""
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    
    # print(geo_transform[0], geo_transform[1], geo_transform[2], geo_transform[3], geo_transform[4], geo_transform[5])
    
    return geo_x, geo_y


def mask_to_polygons(mask_path, mask):
    # Threshold the mask to binary
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    # Create polygons from contours
    # polygons1 = [Polygon(contour[:, 0, :]) for contour in contours if contour.size >= 6]
    for contour in contours:
        if contour.size >= 6:
            listt = []
            for cont in contour[:, 0, :]:
                x, y = pixel_to_geo(mask_path, cont[0], cont[1] )
                arr = np.array([x,y])
                # print(arr[0])
                
                listt.append(arr)
           
            polygons.append(Polygon(listt))
    # print(polygons)
    # print(listt)
    return polygons

def simplify_polygons(polygon_wkts, tolerance):
    """
    Simplify a list of polygons using the Douglas-Peucker algorithm.
    
    :param polygon_wkts: List of polygon WKT strings
    :param tolerance: The tolerance parameter to control the level of simplification
    :return: List of simplified shapely.geometry.Polygon objects
    """
    simplified_polygons = []
    for wkt_str in polygon_wkts:
        polygon = wkt.loads(str(wkt_str))
        simplified_polygon = douglas_peucker(polygon, tolerance)
        simplified_polygons.append(simplified_polygon)
    return simplified_polygons
    

def process_mask_files(mask_folder, output):
    polygons = []

    for mask_filename in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            second = mask_to_polygons(mask_path, mask)
            polygons.extend(second)
            
    #smooothen polygons overlay
    # polygons = simplify_polygons(polygons, 0.0001)
    # polygons_to_tiff(simplified_polygons)
    # print(simplified_polygons)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': polygons})
    
    # Calculate areas and other columns
    gdf = gdf.reset_index()
    gdf['longitude'] = gdf.geometry.centroid.x
    gdf['latitude'] = gdf.geometry.centroid.y
    gdf['taxable'] = 1
    gdf['tax'] = 0
    gdf['type'] = 0
    gdf['value'] = 0
    	

    gdf.to_csv(output+'.csv', index=False)
    
    calculate_area(output+'.csv')
    
    print('saving complete...')
    
    # Optionally, if you have geospatial reference information:
    # Save to shapefile
    # gdf.set_crs(epsg=4326, inplace=True) # Replace 4326 with your EPSG code
    # gdf.to_file('data/test/pred/shp/'+output+'.shp')
    
    
# mask_folder = 'capture/after_processing'
# output = 'capture/final_work/berekuso_buildings'
# process_mask_files(mask_folder, output)


