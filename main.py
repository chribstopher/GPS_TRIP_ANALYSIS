import pandas as pd
from datetime import datetime, timedelta
from GPSPoint import GPSPoint
from gpsParser import GPSParser
import geopandas as gpd
import matplotlib.pyplot as plt
from GPSDataCleaner import GPSDataCleaner as gdc

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

    plot_data(df, no_endpoints)


if __name__ == "__main__":
    parser = GPSParser()
    gps_points = parser.parse_file('gps_files/2025_05_01__145019_gps_file.txt')
    df = parser.to_dataframe()
    clean_dataframe(df)
