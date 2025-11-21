import pandas as pd
import sys
import os
from gpsParser import GPSParser
from GPSDataCleaner import GPSDataCleaner as gdc, GPSDataCleaner
from GPSAnalyzer import GPSAnalyzer
from KMLExporter import KMLExporter
import matplotlib.pyplot as plt

def plot_comparison(original_df, cleaned_df, title="GPS Data Comparison"):
    """Plot original vs cleaned GPS data"""

    # Pull up subplot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original data
    axes[0].plot(original_df['longitude'], original_df['latitude'],
                 'b-', linewidth=1, alpha=0.6)
    # Add start and stop
    axes[0].scatter(original_df['longitude'].iloc[0], original_df['latitude'].iloc[0],
                    c='green', s=100, marker='o', label='Start', zorder=5) # z order 5 means points on top of line
    axes[0].scatter(original_df['longitude'].iloc[-1], original_df['latitude'].iloc[-1],
                    c='red', s=100, marker='s', label='End', zorder=5)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'Original Data ({len(original_df)} points)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cleaned data
    axes[1].plot(cleaned_df['longitude'], cleaned_df['latitude'],
                 'b-', linewidth=1.5, alpha=0.8)
    axes[1].scatter(cleaned_df['longitude'].iloc[0], cleaned_df['latitude'].iloc[0],
                    c='green', s=100, marker='o', label='Start', zorder=5)
    axes[1].scatter(cleaned_df['longitude'].iloc[-1], cleaned_df['latitude'].iloc[-1],
                    c='red', s=100, marker='s', label='End', zorder=5)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title(f'Cleaned Data ({len(cleaned_df)} points)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    # save the image for testing
    plt.savefig(title.replace(' ', '_') + '.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_speed(df, stops_df=None):
    """Plot speed over time with stops marked"""

    fig, ax = plt.subplots(figsize=(15, 5))

    # Convert timestamp to elapsed time in minutes
    df['elapsed_minutes'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

    # Plot speed
    ax.plot(df['elapsed_minutes'], df['speed_knots'], 'b-', linewidth=1.5, label='Speed')

    # Mark stops
    # First check to ensure data has stops
    if stops_df is not None and len(stops_df) > 0:
        # for every stop in row
        for _, stop in stops_df.iterrows():
            # get time stopped
            stop_time = df.iloc[stop['mid_idx']]['elapsed_minutes']
            # plot line
            ax.axvline(x=stop_time, color='r', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Speed (knots)')
    ax.set_title('Speed Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    # save the plot for testing
    plt.savefig('speed_profile.png', dpi=150, bbox_inches='tight')
    plt.show()


def process_gps_file(input_file: str, output_kml: str = None,
                     show_plots: bool = True):
    """
    Complete GPS file processing pipeline

    Args:
        input_file: Path to GPS data file
        output_kml: Path to output KML file (auto-generated if None)
        show_plots: Whether to show matplotlib plots
    """

    # Generate output filename if not provided
    if output_kml is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_kml = f"{base_name}_route.kml"

    # Parse GPS file using GPS parser class
    parser = GPSParser()
    gps_points = parser.parse_file(input_file)

    if not gps_points:
        print("ERROR: No GPS points parsed!")
        return None

    # Create df
    df_original = parser.to_dataframe()

    # Clean data using the GPS data cleaner class
    cleaner = GPSDataCleaner()
    # Remove duplicate points
    df_cleaned = gdc.remove_duplicates(df_original)
    # Remove outliers based on speed / distance
    df_cleaned = gdc.remove_outliers(df_cleaned)
    # Remove idle start and end time
    df_cleaned = gdc.trim_stationary_endpoints(df_cleaned)
    # Remove noise from straight segments
    df_cleaned = gdc.simplify_straight_segments(cleaner, df_cleaned)

    # Analyze data
    analyzer = GPSAnalyzer(df_cleaned)

    # Generate complete trip summary
    summary = analyzer.generate_trip_summary()

    # Get stops and left turns for kml file
    stops_df = summary['stops']
    left_turns_df = summary['left_turns']

    # Create and save the kml file
    # Call KML class to write the file and create kml exporter
    exporter = KMLExporter(df_cleaned, stops_df, left_turns_df)
    # Name the file
    trip_name = f"GPS Track - {summary['start_time'].strftime('%Y-%m-%d %H:%M')}"
    # Create final kml file using exporter
    exporter.generate_kml(output_kml, trip_name)

    # Plot the data
    if show_plots:
        try:
            plot_comparison(df_original, df_cleaned,
                            title=f"GPS Data - {os.path.basename(input_file)}")
            plot_speed(df_cleaned, stops_df)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    return {
        'summary': summary,
        'df_original': df_original,
        'df_cleaned': df_cleaned,
        'output_kml': output_kml
    }




def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Generate KML:  python main.py <gps_file.txt>")
        print("\nOptions:")
        print("  --no-plots    Skip matplotlib visualizations")
        sys.exit(1)

    # Parse arguments
    files = []
    show_plots = True
    use_kalman = False

    for arg in sys.argv[1:]:
        if arg == '--no-plots':
            show_plots = False
        elif not arg.startswith('--'):
            files.append(arg)

    # Process files
    if len(files) == 1:
        # Generate KML with helper function to call classes
        process_gps_file(files[0], show_plots=show_plots)


if __name__ == "__main__":
    main()