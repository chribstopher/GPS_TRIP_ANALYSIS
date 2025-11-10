import pandas as pd
from datetime import datetime, timedelta
from GPSPoint import GPSPoint
from gpsParser import GPSParser
import geopandas as gpd
import matplotlib.pyplot as plt
from GPSDataCleaner import GPSDataCleaner as gdc
import simplekml

def plot_data(df, df2):
    # Create GeoDataFrames
    gdf1 = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    gdf2 = gpd.GeoDataFrame(
        df2, geometry=gpd.points_from_xy(df2.longitude, df2.latitude), crs="EPSG:4326"
    )

    # Create subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot first dataframe
    axes[0].plot(gdf1['latitude'], gdf1['longitude'], color='blue', linewidth=1.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Longitude')
    axes[0].set_title('Latitude')
    axes[0].grid(True)

    # Plot second dataframe
    axes[1].plot(gdf2['latitude'], gdf2['longitude'], color='blue', linewidth=1.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Longitude (SMOOTH)')
    axes[1].set_title('Latitude (SMOOTH)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def clean_dataframe(df):

    no_dupes = gdc.remove_duplicates(df)
    no_outliers = gdc.remove_outliers(no_dupes)
    no_endpoints = gdc.trim_stationary_endpoints(no_outliers)

    # plot_data(df, no_endpoints)

    return no_endpoints

def gps_to_kml(df):

    # create KML object
    kml = simplekml.Kml(open=1)

    # create linestring feature that lies on the ground
    linestring = kml.newlinestring(name="GPS Data")

    # create a list of location tuples (lat lon)
    # zip creates tuples out of data frame columns
    # list creates a list of the location tuples
    location_tuples = list(zip(df['latitude'], df['longitude']))

    # set the coords to locatin tuples
    linestring.coords = location_tuples

    # make the line yellow
    linestring.style.linestyle.color = simplekml.Color.yellow
    linestring.style.linestyle.width = 3

    for idx, row in df.iterrows():
        if row['speed_knots'] <= 0.1:
            point = kml.newpoint()
            point.coords = [(row['longitude'], row['latitude'])]
            point.style.iconstyle.color = simplekml.Color.red

    kml.save("GPS_Linestring.kml")


def visualize_kml():
    # Read the KML file
    # Note: GeoPandas uses fiona which supports KML through the 'KML' driver
    gdf = gpd.read_file('GPS_Linestring.kml', driver='KML')

    # Basic plot
    fig, ax = plt.subplots(figsize=(12, 8))

    lines = gdf[gdf.geometry.type == 'lineString']
    lines.plot(ax=ax, color='yellow', linewidth=3, label='Path')

    points = gdf[gdf.geometry.type == 'Point']
    points.plot(ax=ax, color='red', markersize=5, label='stop')

    plt.title('GPS Track from KML')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    parser = GPSParser()
    gps_points = parser.parse_file('gps_files/2025_05_01__145019_gps_file.txt')
    df = parser.to_dataframe()
    cleaned_df = clean_dataframe(df)

    gps_to_kml(cleaned_df)
    visualize_kml()
