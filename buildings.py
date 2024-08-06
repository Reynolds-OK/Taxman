import csv
import streamlit as st
import folium
from streamlit_folium import st_folium

datafile = "berekuso.csv"


# @st.cache_data
def read_csv():
    data = []
    
    with open(datafile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        print('reading...')
        
        for row in reader:
            longitude, latitude = row['longitude'], row['latitude']
            data.append({
                        'longitude':float(longitude),
                        'latitude':float(latitude)
            })
            
        return data
        

data = read_csv()

BEREKUSO_CENTER = (5.759949211352234, -0.22524781976774413)
map = folium.Map(location=BEREKUSO_CENTER, zoom_start=9)

for building in data:
    location = building['latitude'], building['longitude']
    folium.Marker(location).add_to(map)

st.header('Buildings in Berekuso')
# st.map(data=data, latitude='lat', longitude='long', size=2, color='#0044ff', zoom=14)
st_folium(map, width=1000, zoom=14)

