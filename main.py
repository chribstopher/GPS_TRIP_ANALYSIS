import pandas as pd
import sys
import os
from datetime import datetime
from gpsParser import GPSParser
from GPSDataCleaner import GPSDataCleaner as gdc, GPSDataCleaner
from GPSAnalyzer import GPSAnalyzer
from KMLExporter import KMLExporter
import matplotlib.pyplot as plt
import geopandas as gpd


def plot_comparison(original_df, cleaned_df, title="GPS Data Comparison"):
    """Plot original vs cleaned GPS data"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original data
    axes[0].plot(original_df['longitude'], original_df['latitude'],
                 'b-', linewidth=1, alpha=0.6)
    axes[0].scatter(original_df['longitude'].iloc[0], original_df['latitude'].iloc[0],
                    c='green', s=100, marker='o', label='Start', zorder=5)
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
    plt.savefig(title.replace(' ', '_') + '.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {title.replace(' ', '_')}.png")
    plt.show()


def plot_speed_profile(df, stops_df=None):
    """Plot speed over time with stops marked"""

    fig, ax = plt.subplots(figsize=(15, 5))

    # Convert timestamp to elapsed time in minutes
    df['elapsed_minutes'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 60

    # Plot speed
    ax.plot(df['elapsed_minutes'], df['speed_knots'], 'b-', linewidth=1.5, label='Speed')

    # Mark stops
    if stops_df is not None and len(stops_df) > 0:
        for _, stop in stops_df.iterrows():
            stop_time = df.iloc[stop['mid_idx']]['elapsed_minutes']
            ax.axvline(x=stop_time, color='r', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Speed (knots)')
    ax.set_title('Speed Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('speed_profile.png', dpi=150, bbox_inches='tight')
    print("Saved plot: speed_profile.png")
    plt.show()


def process_gps_file(input_file: str, output_kml: str = None,
                     show_plots: bool = True, use_kalman: bool = False):
    """
    Complete GPS file processing pipeline

    Args:
        input_file: Path to GPS data file
        output_kml: Path to output KML file (auto-generated if None)
        show_plots: Whether to show matplotlib plots
        use_kalman: Whether to apply Kalman filtering
    """

    print("=" * 70)
    print(f"PROCESSING GPS FILE: {input_file}")
    print("=" * 70)

    # Generate output filename if not provided
    if output_kml is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_kml = f"{base_name}_route.kml"

    # STEP 1: Parse GPS file
    print("\n[1/6] Parsing GPS data...")
    parser = GPSParser()
    gps_points = parser.parse_file(input_file)

    if not gps_points:
        print("ERROR: No GPS points parsed!")
        return None

    df_original = parser.to_dataframe()
    print(f"✓ Parsed {len(df_original)} GPS points")

    # STEP 2: Clean data
    cleaner = GPSDataCleaner()
    print("\n[2/6] Cleaning GPS data...")
    df_cleaned = gdc.remove_duplicates(df_original)
    df_cleaned = gdc.remove_outliers(df_cleaned)
    df_cleaned = gdc.trim_stationary_endpoints(df_cleaned)
    df_cleaned = gdc.simplify_straight_segments(cleaner, df_cleaned)


    # Optional: Apply Kalman filtering
    if use_kalman:
        print("  Applying Kalman filter...")
        df_cleaned = gdc.kalman_filtering(df_cleaned)
        print("  ✓ Kalman filtering complete")

    print(f"✓ Cleaned data: {len(df_cleaned)} points remaining")

    # STEP 3: Analyze data
    print("\n[3/6] Analyzing trip data...")
    analyzer = GPSAnalyzer(df_cleaned)

    # Generate complete trip summary
    summary = analyzer.generate_trip_summary()

    # Check for missing time at endpoints
    missing_time = analyzer.estimate_missing_time()

    # STEP 4: Extract stops and turns for KML
    stops_df = summary['stops']
    left_turns_df = summary['left_turns']

    # Optionally get both left and right turns
    # left_turns_df, right_turns_df = analyzer.detect_turns_by_bearing()
    right_turns_df = pd.DataFrame()  # Empty for now

    # STEP 5: Generate KML
    print("\n[4/6] Generating KML file...")
    exporter = KMLExporter(df_cleaned, stops_df, left_turns_df, right_turns_df)

    trip_name = f"GPS Track - {summary['start_time'].strftime('%Y-%m-%d %H:%M')}"
    exporter.generate_kml(output_kml, trip_name)

    # STEP 6: Generate visualizations
    if show_plots:
        print("\n[5/6] Generating visualizations...")
        try:
            plot_comparison(df_original, df_cleaned,
                            title=f"GPS Data - {os.path.basename(input_file)}")
            plot_speed_profile(df_cleaned, stops_df)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    # STEP 7: Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"File: {input_file}")
    print(f"KML Output: {output_kml}")
    print(f"\nTrip Statistics:")
    print(f"  Duration: {summary['duration']} ({summary['duration_minutes']:.2f} minutes)")
    print(f"  Distance: {summary['distance_miles']:.2f} miles")
    print(f"  Avg Speed: {summary['avg_speed_mph']:.1f} mph")
    print(f"  Data Points: {summary['num_points']}")
    print(f"  Stops: {summary['num_stops']} (total: {summary['total_stop_time']:.1f}s)")
    print(f"  Left Turns: {summary['num_left_turns']}")

    if missing_time['missing_start'] or missing_time['missing_end']:
        print(f"\n⚠️  Estimated missing time:")
        if missing_time['missing_start']:
            print(f"    Start: ~{missing_time['estimated_start_time']}s")
        if missing_time['missing_end']:
            print(f"    End: ~{missing_time['estimated_end_time']}s")

    print("=" * 70)

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
        print("  Single file:  python main.py <gps_file.txt>")
        print("  Batch mode:   python main.py <file1.txt> <file2.txt> ...")
        print("\nOptions:")
        print("  --no-plots    Skip matplotlib visualizations")
        print("  --kalman      Apply Kalman filtering")
        print("  --output-dir  Output directory for batch processing (default: 'output')")
        sys.exit(1)

    # Parse arguments
    files = []
    show_plots = True
    use_kalman = False
    output_dir = "output"

    for arg in sys.argv[1:]:
        if arg == '--no-plots':
            show_plots = False
        elif arg == '--kalman':
            use_kalman = True
        elif arg.startswith('--output-dir='):
            output_dir = arg.split('=')[1]
        elif not arg.startswith('--'):
            files.append(arg)

    # Process files
    if len(files) == 1:
        # Single file mode
        process_gps_file(files[0], show_plots=show_plots, use_kalman=use_kalman)


if __name__ == "__main__":
    main()