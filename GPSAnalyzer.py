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

    def detect_left_turns(self, heading_change_threshold=50, speed_threshold=2,
                          window_size=5) -> pd.DataFrame:
        """
        Detect left turns based on heading change

        Args:
            heading_change_threshold: Minimum heading change in degrees to count as turn
            speed_threshold: Minimum speed in knots to consider heading valid
            window_size: Number of points to look ahead for cumulative turn

        Returns:
            DataFrame with columns: [idx, heading_change, latitude, longitude]
        """
        print("\n=== Detecting Left Turns ===")

        turns = []

        # Calculate heading changes between consecutive points
        for i in range(window_size, len(self.df) - window_size):
            curr = self.df.iloc[i]

            # Only consider if vehicle is moving (heading is unreliable when stopped)
            if curr['speed_knots'] < speed_threshold:
                continue

            # Look at heading change over a window to catch gradual turns
            prev_heading = self.df.iloc[i - window_size]['heading']
            curr_heading = curr['heading']

            # Calculate heading change (normalize to -180 to 180)
            heading_change = curr_heading - prev_heading

            # Normalize to -180 to 180 range
            if heading_change > 180:
                heading_change -= 360
            elif heading_change < -180:
                heading_change += 360

            # Negative heading change = left turn (counterclockwise)
            if heading_change < -heading_change_threshold:
                # Check if we already detected a turn nearby (avoid duplicates)
                if not turns or i - turns[-1]['idx'] > window_size * 2:
                    turns.append({
                        'idx': i,
                        'heading_change': heading_change,
                        'latitude': curr['latitude'],
                        'longitude': curr['longitude'],
                        'heading': curr['heading']
                    })

        turns_df = pd.DataFrame(turns)
        print(f"Detected {len(turns_df)} left turns (change >= {heading_change_threshold}°)")

        if len(turns_df) > 0:
            print(f"Average turn angle: {abs(turns_df['heading_change'].mean()):.1f}°")

        return turns_df

    def detect_turns_by_bearing(self, angle_threshold=45, speed_threshold=2,
                                window_size=3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Alternative turn detection using calculated bearing between points
        More reliable than GPS heading when available

        Returns:
            Tuple of (left_turns_df, right_turns_df)
        """
        print("\n=== Detecting Turns by Bearing ===")

        # Calculate bearing for each segment
        bearings = []
        for i in range(len(self.df) - 1):
            curr = self.df.iloc[i]
            next_pt = self.df.iloc[i + 1]

            bearing = get_curdirection(
                curr['latitude'], curr['longitude'],
                next_pt['latitude'], next_pt['longitude']
            )
            bearings.append(bearing)

        bearings.append(bearings[-1])  # Duplicate last bearing
        self.df['bearing'] = bearings

        left_turns = []
        right_turns = []

        for i in range(window_size, len(self.df) - window_size):
            curr = self.df.iloc[i]

            # Only consider if moving
            if curr['speed_knots'] < speed_threshold:
                continue

            # Calculate bearing change over window
            prev_bearing = self.df.iloc[i - window_size]['bearing']
            curr_bearing = curr['bearing']

            bearing_change = curr_bearing - prev_bearing

            # Normalize to -180 to 180
            if bearing_change > 180:
                bearing_change -= 360
            elif bearing_change < -180:
                bearing_change += 360

            # Detect significant turns
            if bearing_change < -angle_threshold:
                # Left turn
                if not left_turns or i - left_turns[-1]['idx'] > window_size * 2:
                    left_turns.append({
                        'idx': i,
                        'bearing_change': bearing_change,
                        'latitude': curr['latitude'],
                        'longitude': curr['longitude']
                    })
            elif bearing_change > angle_threshold:
                # Right turn
                if not right_turns or i - right_turns[-1]['idx'] > window_size * 2:
                    right_turns.append({
                        'idx': i,
                        'bearing_change': bearing_change,
                        'latitude': curr['latitude'],
                        'longitude': curr['longitude']
                    })

        left_df = pd.DataFrame(left_turns)
        right_df = pd.DataFrame(right_turns)

        print(f"Detected {len(left_df)} left turns, {len(right_df)} right turns")

        return left_df, right_df

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