import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple
from GPSDataCleaner import get_curdirection


class GPSAnalyzer:
    """Analyze GPS data for stops, turns, and trip statistics"""

    def __init__(self, df: pd.DataFrame):
        """Initialize with cleaned GPS DataFrame"""
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def detect_stops(self, speed_threshold=0.5, min_duration=3) -> pd.DataFrame:
        """
        Detect stops (traffic lights, stop signs, etc.)

        Args:
            speed_threshold: Speed in knots below which vehicle is considered stopped
            min_duration: Minimum duration in seconds to count as a stop

        Returns:
            DataFrame with columns: [start_idx, end_idx, duration, latitude, longitude]
        """
        print("\n=== Detecting Stops ===")

        stops = []
        i = 0

        while i < len(self.df):
            # Check if vehicle is stopped
            if self.df.iloc[i]['speed_knots'] <= speed_threshold:
                stop_start_idx = i
                stop_start_time = self.df.iloc[i]['timestamp']

                # Find end of stop
                while i < len(self.df) and self.df.iloc[i]['speed_knots'] <= speed_threshold:
                    i += 1

                stop_end_idx = i - 1
                stop_end_time = self.df.iloc[stop_end_idx]['timestamp']

                # Calculate duration
                duration = (stop_end_time - stop_start_time).total_seconds()

                # Only count stops longer than minimum duration
                if duration >= min_duration:
                    # Get location (use middle of stop for better accuracy)
                    mid_idx = (stop_start_idx + stop_end_idx) // 2
                    stops.append({
                        'start_idx': stop_start_idx,
                        'end_idx': stop_end_idx,
                        'mid_idx': mid_idx,
                        'duration': duration,
                        'latitude': self.df.iloc[mid_idx]['latitude'],
                        'longitude': self.df.iloc[mid_idx]['longitude']
                    })
            else:
                i += 1

        stops_df = pd.DataFrame(stops)
        print(f"Detected {len(stops_df)} stops (duration >= {min_duration}s)")

        if len(stops_df) > 0:
            print(f"Total stop time: {stops_df['duration'].sum():.1f} seconds")

        return stops_df

    def detect_left_turns(self, heading_change_threshold=3e-07, speed_threshold=2,
                          window_size=5) -> pd.DataFrame:
        """
        Detect left turns based on heading change

        Args:'
            heading_change_threshold: Minimum heading change in degrees to count as turn, 2e-7 chosen based on all
            cross products calculated in gps data
            speed_threshold: Minimum speed in knots to consider heading valid
            window_size: Number of points to look ahead for cumulative turn

        Returns:
            DataFrame with columns: [idx, heading_change, latitude, longitude]
        """
        turns = []

        print("_______detecting left turns__________")

        # Calculate heading changes between consecutive points
        for i in range(window_size, len(self.df) - window_size):
            curr = self.df.iloc[i]

            # Only consider if vehicle is moving
            if curr['speed_knots'] < speed_threshold:
                continue

            # calculate left turn angle using z-component cross product
            # which needs 3 points
            prev = self.df.iloc[i - window_size]
            next = self.df.iloc[i + window_size]

            vector1 = np.array([
                curr['longitude'] - prev['longitude'],
                curr['latitude'] - prev['latitude']
            ])

            vector2 = np.array([
                next['longitude'] - curr['longitude'],
                next['latitude'] - curr['latitude']
            ])

            # get 2d cross product
            cross_prod = vector1[0] * vector2[1] - vector1[1] * vector2[0]

            # print(f"Index {i}: cross_prod = {cross_prod}")

            # Calculate magnitudes
            mag_v1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
            mag_v2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

            # Avoid division by zero
            if mag_v1 > 0 and mag_v2 > 0:
                # Normalize cross product
                normalized_cross = cross_prod / (mag_v1 * mag_v2)

            # check if turn is left (negative z component)
            if normalized_cross < -0.5:
                # check for a nearby turn to avoid duplicates
                if not turns or i - turns[-1]['idx'] > window_size * 2:
                    turns.append({
                        'idx': i,
                        'cross_product_z': cross_prod,
                        'latitude': curr['latitude'],
                        'longitude': curr['longitude'],
                    })

        # create turns DF and return
        turns_df = pd.DataFrame(turns)
        return turns_df

    def calculate_trip_duration(self) -> Tuple[timedelta, float, float]:
        """
        Calculate total trip duration from first to last point

        Returns:
            Tuple of (timedelta, minutes, seconds)
        """
        if len(self.df) < 2:
            return timedelta(0), 0.0, 0.0

        start_time = self.df.iloc[0]['timestamp']
        end_time = self.df.iloc[-1]['timestamp']

        duration = end_time - start_time
        minutes = duration.total_seconds() / 60
        seconds = duration.total_seconds()

        print(f"\n=== Trip Duration ===")
        print(f"Start: {start_time}")
        print(f"End: {end_time}")
        print(f"Duration: {duration}")
        print(f"Minutes: {minutes:.2f}")

        return duration, minutes, seconds

    def calculate_distance(self) -> Tuple[float, float]:
        """
        Calculate total distance traveled
        Uses haversine distance between consecutive points

        Returns:
            Tuple of (distance_meters, distance_miles)
        """
        from GPSDataCleaner import haversine_distance

        if len(self.df) < 2:
            return 0.0, 0.0

        total_distance_m = 0.0

        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            distance = haversine_distance(
                prev['latitude'], prev['longitude'],
                curr['latitude'], curr['longitude']
            )
            total_distance_m += distance

        distance_miles = total_distance_m / 1609.34

        print(f"\n=== Distance ===")
        print(f"Total: {total_distance_m:.1f} meters ({distance_miles:.2f} miles)")

        return total_distance_m, distance_miles

    def calculate_average_speed(self) -> Tuple[float, float]:
        """
        Calculate average moving speed

        Returns:
            Tuple of (avg_speed_knots, avg_speed_mph)
        """
        # Only consider points where vehicle is moving
        moving_df = self.df[self.df['speed_knots'] > 0.5]

        if len(moving_df) == 0:
            return 0.0, 0.0

        avg_knots = moving_df['speed_knots'].mean()
        avg_mph = avg_knots * 1.15078

        print(f"\n=== Average Speed (moving) ===")
        print(f"{avg_knots:.2f} knots ({avg_mph:.2f} mph)")

        return avg_knots, avg_mph

    def generate_trip_summary(self) -> dict:
        """
        Generate comprehensive trip summary

        Returns:
            Dictionary with all trip statistics
        """
        print("\n" + "=" * 50)
        print("TRIP SUMMARY")
        print("=" * 50)

        # Basic stats
        duration, minutes, seconds = self.calculate_trip_duration()
        distance_m, distance_mi = self.calculate_distance()
        avg_knots, avg_mph = self.calculate_average_speed()

        # Detect features
        stops_df = self.detect_stops()
        left_turns_df = self.detect_left_turns()

        # Create summary dictionary
        summary = {
            'start_time': self.df.iloc[0]['timestamp'],
            'end_time': self.df.iloc[-1]['timestamp'],
            'duration': duration,
            'duration_minutes': minutes,
            'duration_seconds': seconds,
            'distance_meters': distance_m,
            'distance_miles': distance_mi,
            'avg_speed_knots': avg_knots,
            'avg_speed_mph': avg_mph,
            'num_points': len(self.df),
            'num_stops': len(stops_df),
            'num_left_turns': len(left_turns_df),
            'total_stop_time': stops_df['duration'].sum() if len(stops_df) > 0 else 0,
            'stops': stops_df,
            'left_turns': left_turns_df
        }

        print("\n" + "=" * 50)
        return summary

    def estimate_missing_time(self, expected_start_moving=True,
                              expected_end_moving=True,
                              avg_city_speed_mph=25) -> dict:
        """
        Estimate missing time if GPS started/stopped mid-journey

        Args:
            expected_start_moving: True if car should have been moving at start
            expected_end_moving: True if car should have been moving at end
            avg_city_speed_mph: Assumed average speed for estimation

        Returns:
            Dictionary with estimated missing time
        """
        print("\n=== Estimating Missing Time ===")

        first_point = self.df.iloc[0]
        last_point = self.df.iloc[-1]

        estimates = {
            'missing_start': False,
            'missing_end': False,
            'estimated_start_time': 0,
            'estimated_end_time': 0
        }

        # Check if started while moving
        if expected_start_moving and first_point['speed_knots'] > 2:
            print(f"⚠️  GPS started mid-journey (speed: {first_point['speed_knots']:.1f} knots)")
            # Estimate time to accelerate from 0 to current speed
            # Assume ~10 seconds to reach cruising speed
            estimates['missing_start'] = True
            estimates['estimated_start_time'] = 10
            print(f"   Estimated missing start time: ~{estimates['estimated_start_time']}s")

        # Check if ended while moving
        if expected_end_moving and last_point['speed_knots'] > 2:
            print(f"⚠️  GPS ended mid-journey (speed: {last_point['speed_knots']:.1f} knots)")
            # Estimate time to decelerate
            estimates['missing_end'] = True
            estimates['estimated_end_time'] = 10
            print(f"   Estimated missing end time: ~{estimates['estimated_end_time']}s")

        total_estimated = estimates['estimated_start_time'] + estimates['estimated_end_time']
        if total_estimated > 0:
            print(f"\nTotal estimated missing time: ~{total_estimated}s")
        else:
            print("✓ Trip appears complete (started and ended stationary)")

        return estimates