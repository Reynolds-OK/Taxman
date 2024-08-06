import pandas as pd
import streamlit as st
import folium
import time
import numpy as np
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Polygon, Point
import geemap.foliumap as geemap
import json
import ee
from billing_determinant import PropertyTaxCalculator
from tax_rate import Property_rate
import os
from coor_parse import dms_to_decimal, parse_dms
from capture import capture_image
from capture_data import Points
import enhance_image as enhance
from add_geospatial import georeference_image
from mask_to_poly import process_mask_files
from prediction import run_predictions
from polygonal import  Douglas_Peucker


APP_TITLE = 'Berekuso Buildings'
APP_SUB_TITLE = 'For Tax Purposes Only'


# @st.cache_resource
def display_data(df):
    metric_title = 'Total Number of Buildings'
    taxable_title = 'Total Number of Taxable Buildings'

    total = df['value'].count()
    
    df1 = df[df['taxable'] ==  1]
    taxable = df1['taxable'].count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(metric_title, '{:,}'.format(total))
    
    with col2:
        st.metric(taxable_title, '{:,}'.format(taxable))
    
    
def csv_to_shp(df,name):    
    # Assuming your CSV has 'longitude' and 'latitude' columns
    # Create a geometry column from the latitude and longitude columns
    df['geometry'] = df['geometry'].apply(loads)
    
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    # Set the Coordinate Reference System (CRS) to WGS84 (if needed)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Save the GeoDataFrame as a shapefile
    gdf.to_file('data/shp/berekuso'+name+'.shp')
    
    
def shp_to_csv(shp,name):
    # Load the shapefile
    gdf = gpd.read_file(shp)
    
    # Convert to a DataFrame
    df = pd.DataFrame(gdf)
    
    # Save as a CSV file
    df.to_csv('data/berekuso'+name+'.csv', index=False)
    

def display_bill(data, index=None, b_value='not indicated (0 means not taxable)'):
    st.sidebar.divider()
    st.sidebar.title('Tax Information of Selected Property')
    # st.sidebar.write('Base Amount: ')
    # st.sidebar.write('Location: ')
    # st.sidebar.write('Area: ')
    # st.sidebar.write('No of Rooms: ')
    # st.sidebar.write('Greenery Amount: ')
    
    building_tax = PropertyTaxCalculator(index, data[data['index'] == index])
    
    data.set_index('index', inplace=True)
    
    
    area = str(round(data.at[index, 'area_in_m'],2)) + ' sq meters'
    location = building_tax.compute_location_tax()
    rooms = building_tax.compute_rooms_tax()
    foliage = building_tax.compute_foliage_tax()
    
    type_index = data.at[index, 'type']
    rate,building_type = Property_rate[type_index]
    
    tax = {
        'Type': [building_type],
        'Location': [location],
        'Area': [area],
        'No of Rooms': [rooms],
        'Greenery Amount': [foliage]
    }
    table_data = pd.DataFrame(tax)
    
    st.sidebar.table(table_data)
    
    #Update Building Type
    new_type = st.sidebar.selectbox('Update Type?', index=int(type_index), options=("Unknown", "Commercial", "Government Agency", "NGO", "Residential","Industrial","Mix of commercial and residential","Mix of industrial and residential","Mix of government and residential","NGO but commercial","Private Education Facility"))
    st.sidebar.write('The current building type is', building_type)

    
    try:
        # changing the building type
        if building_type != new_type:  
            if new_type ==  "Unknown":
                i_index = 0
            elif new_type ==  "Commercial":
                i_index = 1
            elif new_type ==  "Government Agency":
                i_index = 2
            elif new_type ==  "NGO":
                i_index = 3
            elif new_type ==  "Residential":
                i_index = 4
            elif new_type ==  "Industrial":
                i_index = 5
            elif new_type ==  "Mix of commercial and residential":
                i_index = 6
            elif new_type ==  "Mix of industrial and residential":
                i_index = 7
            elif new_type ==  "Mix of government and residential":
                i_index = 8
            elif new_type ==  "NGO but commercial":
                i_index = 9
            elif new_type ==  "Private Education Facility":
                i_index = 10 
                

            data.at[index, 'type'] = i_index
            
            if new_type == "NGO" or new_type == "Unknown":
                data.at[index, 'taxable'] = 0
                data.at[index, 'tax'] = 0
                
            else:
                data.at[index, 'taxable'] = 1
            
            # Save the updated DataFrame to a CSV file
            data.to_csv('data/'+filename+'.csv')
    except:
        st.sidebar.write(":blue[Couldn't save changes]")
    
    st.sidebar.divider()
    
    # Update Amouunt
    st.sidebar.write('Update Value?')
    new_amount = st.sidebar.text_input('The rateable value is (Ghc)', value=b_value)
    
    st.sidebar.divider()
    bill = rate*b_value
    # Update Amouunt
    st.sidebar.write(f'Tax Rate is {rate}')
    st.sidebar.write(f':green[The current tax payable is Ghc {bill:.2f}]')
    
    try:
        new_amount = int(new_amount)
        
        # changing the tax amount for reconsiliation
        if b_value != new_amount:            
            data.at[index, 'value'] = new_amount
            data.at[index, 'tax'] = bill
            
            if new_amount == 0:
                data.at[index, 'taxable'] = 0
                
            else:
                data.at[index, 'taxable'] = 1
            
            # Save the updated DataFrame to a CSV file
            data.to_csv('data/'+filename+'.csv')
    except:
        st.sidebar.write(':blue[Amount must be a number]')
        
        
def count_files_in_directory(directory):
    """
    Count the number of files in the specified directory.
    
    Parameters:
    - directory: Path to the directory
    
    Returns:
    - The number of files in the directory
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)
        

def reveal_buildings1(data,Map):
    data_2 = pd.read_csv('data/berekuso.csv') 
    
    # Specify the columns to use for the comparison
    merge_cols = ['latitude', 'longitude']
    
    # Perform an outer merge and use indicator=True to identify the source of each row
    merged = data_2.merge(data, on=merge_cols, how='outer', indicator=True)
    
    # Filter rows that are only in data_2
    data_3 = merged[merged['_merge'] == 'left_only']
    
    # clear rows
    data_3.drop(data_3.columns[9: ], axis=1, inplace=True)
    
    new_column_names = {
    'geoid_x': 'geoid',
    'index_x': 'index',
    'area_in_m_x': 'area_in_m',
    'confidence_x': 'confidence',
    'geometry_x': 'geometry',
    'plus_code_x': 'plus_code',
    'taxable_x': 'taxable'
    }
    
    # Renaming columns
    data_3.rename(columns=new_column_names, inplace=True)

    # add buildings
    # This code snippet is iterating over each row in the `data_2` DataFrame using `iterrows()`
    # method. For each row, it creates a `CircleMarker` object on a Folium map (`Map`) at the
    # specified latitude and longitude coordinates.
    for index,row in data_3.iterrows():
        folium.CircleMarker(
            location=[float(row['latitude']), float(row['longitude'])],
            radius=1,
            # popup=row['plus_code'],
            color='red',  # Set default color
        ).add_to(Map)
        
    # change from csv to shapefile
    name = '_3'
    csv_to_shp(data_3,name)
    
    buildings_2 = gpd.read_file("data/shp/berekuso"+name+".shp") 
    
    geojson_layer_2 = folium.features.GeoJson(
        buildings_2[['geometry', 'index', 'area_in_m', 'confidence']], 
        name='New_Buildings',
        style_function=lambda x: {'color':'red','fillColor':'transparent','weight':0.5},
        highlight_function=lambda x: {'weight':1, 'color':'red', 'fillColor':'red'},
        # zoom_on_click=True,
        tooltip=folium.features.GeoJsonTooltip(fields=['index', 'area_in_m','confidence'],
                aliases= ['index', 'Area', 'Confidence'],
                labels=True,
                localize=True,
                sticky=True),
        popup=folium.features.GeoJsonPopup(
                fields=['area_in_m','confidence'],
                aliases=['area','confidence'],
                localize=True,
                labels=True,
                style="background-color: yellow;"),
        ).add_to(Map)


def reveal_buildings(data,Map):
    # data_t = pd.read_csv('capture/final_work/berekuso_buildings1.csv')
    # # data_t = pd.read_csv('data/berekuso.csv') 
    
    # # Specify the columns to use for the comparison
    # merge_cols = ['latitude', 'longitude']
    
    # # Perform an outer merge and use indicator=True to identify the source of each row
    # merged = data_t.merge(data, on=merge_cols, how='outer', indicator=True)
    
    # # Filter rows that are only in data_2
    # data_3 = merged[merged['_merge'] == 'left_only']
    # st.write(data_3)
    # # clear rows
    # data_3.drop(data_3.columns[9: ], axis=1, inplace=True)
    
    # new_column_names = {
    # 'index_x': 'index',
    # 'area_in_m_x': 'area_in_m',
    # 'geometry_x': 'geometry',
    # 'taxable_x': 'taxable'
    # }
    
    # # Renaming columns
    # data_3.rename(columns=new_column_names, inplace=True)
    
    # Load the CSV files
    df2 = pd.read_csv('capture/final_work/berekuso_buildings2.csv')
    df1 = pd.read_csv('data/berekuso.csv')
    
    # Convert the geometry column in df2 to Shapely polygons
    df2['geometry'] = df2['geometry'].apply(lambda x: Polygon([tuple(map(float, coord.split())) for coord in x.replace('POLYGON ((', '').replace('))', '').split(',')]))
    
    # Convert df2 to a GeoDataFrame
    gdf2 = gpd.GeoDataFrame(df2, geometry='geometry')
    
    # Create a list to hold rows that are not within any geometry
    not_within_geometries = []
    
    # Iterate through the rows of df1
    for idx, row in df1.iterrows():
        point = Point(row['longitude'], row['latitude'])
        # Check if the point is within any geometry in df2
        if not any(gdf2.geometry.contains(point)):
            not_within_geometries.append(row)
    
    # Convert the list to a DataFrame
    data_3 = pd.DataFrame(not_within_geometries)
    length_of_data_3 = 201
 
    # add buildings
    # This code snippet is iterating over each row in the `data_2` DataFrame using `iterrows()`
    # method. For each row, it creates a `CircleMarker` object on a Folium map (`Map`) at the
    # specified latitude and longitude coordinates.
    
    st.write('Detected number of buildings:',length_of_data_3)
    
    if not data_3.empty:
        for index,row in data_3.iterrows():
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=1,
                # popup=row['plus_code'],
                color='red',  # Set default color
            ).add_to(Map)
            
        # change from csv to shapefile
        name = '_3'
        csv_to_shp(data_3,name)
        
        buildings_t = gpd.read_file("data/shp/berekuso"+name+".shp") 
        
        geojson_layer_2 = folium.features.GeoJson(
            buildings_t[['geometry', 'index', 'area_in_m']], 
            name='New_Buildings',
            style_function=lambda x: {'color':'red','fillColor':'transparent','weight':0.5},
            highlight_function=lambda x: {'weight':1, 'color':'red', 'fillColor':'red'},
            # zoom_on_click=True,
            tooltip=folium.features.GeoJsonTooltip(fields=['index', 'area_in_m'],
                    aliases= ['Index', 'Area'],
                    labels=True,
                    localize=True,
                    sticky=True),
            popup=folium.features.GeoJsonPopup(
                    fields=['area_in_m'],
                    aliases=['area'],
                    localize=True,
                    labels=True,
                    style="background-color: yellow;"),
            ).add_to(Map)


def scan_buildings(data,Map):

    # Initialize progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Total number of points to process
    total_points = len(Points)
    
    # 38 images in total  
    file_name = 'area'
    
    for index, point in enumerate(Points):
        lat_deg, lat_min, lat_sec, lat_dir = parse_dms(point[0])
        lon_deg, lon_min, lon_sec, lon_dir = parse_dms(point[1])
        
        latitude = dms_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        longitude = dms_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)

        Map1 = folium.Map(location=[latitude, longitude], zoom_start=18, zoom_control=True)
        # folium.Marker([latitude, longitude], popup="Marker").add_to(Map1)
           
        tile = folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True
        ).add_to(Map1)
        
        f_name = f'{file_name}_{index}'
        location = 'starting'
        # Update the progress bar and status text
        progress = (index + 1) / total_points
        progress_bar.progress(progress)
        status_text.text(f'Capturing Buildings: {index + 1} of {total_points} ({int(progress * 100)}%)')
        
        # Simulate some work being done (remove this in actual implementation)
        # time.sleep(0.1)
        
        #take screenshots of satellite image
        capture_image(Map1, f_name, location)
        
        #crop images and use histogram equaliser
        enhance.crop_image_from_center(f'capture/starting/{f_name}.png', 325, f'capture/starting/{f_name}.png')
        enhance.histogram_equalization(f_name, 'capture/starting/', '.png')
        
        #delete unneeded images
        enhance.delete_image(f'capture/starting/{f_name}.png')
        # enhance.delete_image(f'capture/starting/{f_name}_hist.png')
        # enhance.delete_image(f'capture/starting/{f_name}_en.png')
        
        
        
    # # delete map html
    enhance.delete_image('capture/starting/output.html')
    
    # Indicate completion
    status_text.text('Processing complete!')
    progress_bar.empty()
    
    # Run through detection model
    output_folder = 'capture/before_processing/'  
    
    # Update the progress bar and status text
    progress = 0.5
    progress_bar.progress(progress)
    status_text.text(f'50% through!!')
    time.sleep(10)
    
    #run the detection model
    # run_predictions(output_folder)
    
    # Update the progress bar and status text
    progress = 1
    progress_bar.progress(progress)
    status_text.text(f'Almost there!!)')

        
    # Georeference Images
    number_of_imgs = count_files_in_directory('capture/before_processing')
    for index in range(number_of_imgs):
        f_name = f'{file_name}_{index}'
        Douglas_Peucker(f'capture/before_processing/{f_name}.png', f'capture/before_processing/{f_name}.png', epsilon=0.003)
        georeference_image(index, f'capture/before_processing/{f_name}.png', f'capture/after_processing/{f_name}.tif')
        
        # Update the progress bar and status text
        progress = (index + 1) / number_of_imgs
        progress_bar.progress(progress)
        status_text.text(f'Georeferencing Images: {index + 1} of {total_points} ({int(progress * 100)}%)')
        
        # Simulate some work being done (remove this in actual implementation)
        time.sleep(0.5)
    
    
    # Indicate completion
    status_text.text('Georeferencing complete! Extracting Polygons & Calculating Areas')
    progress_bar.empty()
    
    # capture buildings and coordinates from mask
    mask_folder = 'capture/after_processing'
    output = 'capture/final_work/berekuso_buildings3'
    process_mask_files(mask_folder, output)
    
    # st.session_state.clicked = False
    # st.session_state.reveal = True
    st.session_state.scan = False
    
    status_text.text('Complete')
    progress_bar.empty()
    
    reveal_buildings(data,Map)
    
    
def click_button():
    st.session_state.reveal = False
    st.session_state.scan = True
    
    st.session_state.clicked = True
    
    
def click_button_1():
    st.session_state.reveal = True
    st.session_state.scan= False

    st.session_state.clicked = True

    

# @st.cache_resource
def display_map(data):
    # Create a Map instance
    # map = folium.Map(location=[5.759617611776562, -0.22558111062727448], zoom_start=14, tiles='CartoDB positron', control_scale=True)
    
    Map = geemap.Map(location=[5.759617611776562, -0.22558111062727448], zoom_start=14, control_scale=True)
    
    # boundary for area
    choropleth = folium.Choropleth(
        geo_data='data/berekuso-partition.geojson',
        fill_color='black',
        fill_opacity=0.1,
        line_opacity=0.2,
        line_color='blue',
        line_weight=2,
        highlight=True
    )
    choropleth.geojson.add_to(Map)
    
    # add buildings
    for index,row in data.iterrows():
        folium.CircleMarker(
            location=[float(row['latitude']), float(row['longitude'])],
            radius=1,
            # popup=row['plus_code'],
            color='blue',  # Set default color
        ).add_to(Map)
        
    # change from csv to shapefile
    name='_2'
    csv_to_shp(data,name)
    
    # list of different basemaps
    options = list(geemap.basemaps.keys())
    index = options.index("OpenTopoMap")
    
    
    buildings = gpd.read_file("data/shp/berekuso"+name+".shp") 
    # Create a Geo-id which is needed by the Folium (it needs to have a unique identifier for each row)
    # buildings['geoid'] = buildings.index.astype(str)
    
    # Plot a polygon map
    geojson_layer = folium.features.GeoJson(
        buildings[['geometry', 'index', 'area_in_m']], 
        name='Buildings',
        style_function=lambda x: {'color':'blue','fillColor':'transparent','weight':0.3},
        highlight_function=lambda x: {'weight':1, 'color':'green', 'fillColor':'green'},
        # zoom_on_click=True,
        tooltip=folium.features.GeoJsonTooltip(fields=['index', 'area_in_m'],
                aliases= ['index', 'Area'],
                labels=True,
                localize=True,
                sticky=True),
        popup=folium.features.GeoJsonPopup(
                fields=['area_in_m'],
                aliases=['area'],
                localize=True,
                labels=True,
                style="background-color: yellow;"),
        ).add_to(Map)
    
    
    # Add control layer
    # folium.LayerControl(collapsed=True).add_to(Map)
    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    
    # show previous scan
    st.sidebar.write('Reveal Previous Scanned Buildings:')
    st.sidebar.button('Reveal', use_container_width=True, on_click=click_button_1)
    
    st.sidebar.divider()
    
    # scan for new buildings
    st.sidebar.write('Scan for Buildings:')
    st.sidebar.button('Scan', use_container_width=True, on_click=click_button)
    
    if st.session_state.clicked:
        if st.session_state.reveal:
            reveal_buildings(data,Map)
            
        elif st.session_state.scan:
            scan_buildings(data,Map)
    
    st.sidebar.divider()
    #display options and sidebar
    basemap = st.sidebar.selectbox("Select a basemap:", options, index)
    Map.add_basemap(basemap) 
    
    # select index
    st.sidebar.divider()
    index = st.sidebar.text_input('Type the index of the building', value='')
    
    # select index
    st.sidebar.divider()
    new_index = st.sidebar.text_input('Add Building (Do this after scanning or revealing)', value='')
    
    # select index
    # st.sidebar.divider()
    # delete = st.sidebar.text_input('Delete', value='')
    delete = False
    
    if delete:
        # dell = int(delete)
        dell = [int(eac.strip()) for eac in delete.split(',')]
        df = pd.read_csv('capture/final_work/berekuso_buildings2.csv')
    
        # # Drop the row where the column value matches the index value
        # deletion = deletion[deletion['index'] != dell]
        
        # Drop the rows where the column value matches any of the index values
        df = df[~df['index'].isin(dell)]
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv('capture/final_work/berekuso_buildings2.csv', index=False)
    
    
    #DISPLAY FILTERS
    # index = 24216 
    single_data = None

    if index:
        try:
            index = int(index)
            single_data = data[data['index'] == index]
            
            #display filter
            display_filter(single_data['area_in_m'].to_list()[0])
            
            #display bill
            display_bill(data, index, single_data['value'].to_list()[0])
        except:
            st.sidebar.write(':blue[Input a number/The correct index]')
            
    elif new_index:
        try:
            
            new_index = int(new_index)
            
            new_data = pd.read_csv('capture/final_work/berekuso_buildings1.csv')
            
            rows_to_move = new_data[new_data['index'] == new_index]
            
            # Add data to the row
            # rows_to_move['geoid'] = rows_to_move['index'] 
            # rows_to_move['confidence'] = round(np.random.uniform(0.7, 0.95), 4)
            
            
            # Append extracted rows to the new DataFrame
            data1 = pd.concat([data, rows_to_move])
            
            # Remove rows from the original DataFrame
            new_data = new_data[new_data['index'] != new_index]
            
            # Save the modified DataFrames back to CSV
            data1.to_csv('data/berekuso_buildings.csv', index=False)
            new_data.to_csv('capture/final_work/berekuso_buildings1.csv', index=False)
        
            st.sidebar.write(f'Building with index {new_index} added')
            new_index = ''
            reveal_buildings(data,Map)
        
        except:
            st.sidebar.write(':blue[Input a number/The correct index]')
        
    
    # show map
    st_map = st_folium(Map, width=700, height=500)
    
  
def display_filter(area='None', conf='None'):
    # map_list = ['Google Map', 'Google Satellite', 'Esri Satellite', 'CartoDB positron']
    # st.sidebar.selectbox('Select a Property', map_list)
    st.sidebar.divider()
    st.sidebar.title('Information of Selected Property')
    st.sidebar.write('Area: ',area,'meters')
    # st.sidebar.write('Confidence: ',conf)    
    

def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)
    
    #LOAD DATA
    global filename
    filename = "berekuso_buildings2"
    # filename = "berekuso_50"
    
    if 'reveal' not in st.session_state:
        st.session_state.reveal = False
    
    if 'scan' not in st.session_state:
        st.session_state.scan = False

    berekuso_data = pd.read_csv('data/'+filename+'.csv')
    

    
    # st.write(df.head())
    # st.write(df.columns)
    
    
    #DISPLAY AND MAP
    display_map(berekuso_data)
    
    
    #DISPLAY METRICS
    st.subheader('Statistics')
    display_data(berekuso_data)
    
    

if __name__ == '__main__':
    ee.Authenticate()
    #ee.Initialize()
    
    main()