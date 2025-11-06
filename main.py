import pandas as pd
from datetime import datetime, timedelta
from GPSPoint import GPSPoint
from gpsParser import GPSParser
import geopandas as gpd
import matplotlib.pyplot as plt
from GPSDataCleaner import GPSDataCleaner as gdc

def plot_data(df, df2):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(gdf['latitude'], gdf['longitude'], color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('longitude')
    plt.title('latitude')
    plt.grid(True)

    gdf = gpd.GeoDataFrame(
        df2, geometry=gpd.points_from_xy(df2.longitude, df2.latitude), crs="EPSG:4326"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(gdf['latitude'], gdf['longitude'], color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('longitude SMOOTH')
    plt.title('latitude SMOOTH')
    plt.grid(True)
    plt.show()


def clean_dataframe(df):

    no_dupes = gdc.remove_duplicates(df)
    no_outliers = gdc.remove_outliers(no_dupes)


if __name__ == "__main__":
    parser = GPSParser()
    gps_points = parser.parse_file('gps_files/2025_05_01__145019_gps_file.txt')
    df = parser.to_dataframe()
    clean_dataframe(df)
