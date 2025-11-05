from datetime import datetime
from typing import List


class GPSDataCleaner:
    """
    A class to clean and preprocess GPS data.
    """

    def __init__(self, points: List[GPSPoint]):
        self.points = points

    def remove_duplicates(self, time_threshold_seconds=0.1) -> List[GPSPoint]:
        """
        Remove duplicate GPS points that are within a certain time threshold.
        """
        if not self.points:
            return []

        cleaned = [self.points[0]]
        for point in self.points[1:]:
            last = cleaned[-1]
            time_diff = abs((point.timestamp - last.timestamp).total_seconds())

            # if the points are very close in time and location, skip the duplicate
            if (time_diff < time_threshold_seconds and 
                abs(point.latitude - last.latitude) < 0.0001 and
                abs(point.longitude - last.longitude) < 0.0001):
                continue
            cleaned.append(point)
        print(f"Removed {len(self.points) - len(cleaned)} duplicate points.")
        return cleaned

    def remove_outliers(self, points: List[GPSPoint], max_speed_knots=100) -> List[GPSPoint]:
        """
        Removes GPS points that represent impossible movements (Gps glitches)

        CHECK THE CHECKSUMS AT SOME POINT?
        """
        if len(points) < 2:
            return points

        cleaned = [points[0]]
        for i in range(1, len(points)):
            prev = cleaned[-1]
            curr = points[i]
            # calculating the time difference
            time_diff = (curr.timestamp - prev.timestamp).total_seconds()
            if time_diff <= 0:
                continue  # skip invalid time differences
            
            # calculating the distance
            distance = prev.distance_to(curr)

            #calculate implied speed in knots
            implied_speed_knots = (distance / time_diff) * 1.94384  # meters per second to knots

            # if the speed is impossibly high, skip the point
            if implied_speed_knots > max_speed_knots:
                continue

            # check for very large jumps in location that could signify errors
            if abs(curr.latitude - prev.latitude) > 10 or abs(curr.longitude - prev.longitude) > 10:
                continue

            cleaned.append(curr)
        print(f"Removed {len(points) - len(cleaned)} outlier points.")
        return cleaned    

    def trim_stationary_endpoints(self, points: List[GPSPoint], 
        movement_threshold_knots=0.5) -> Tuple[List[GPSPoint], int, int]:
        
        """ Removes stationary points from the start and end of the trip"""
        # find first moving point
        start_idx = 0
        for i, point in enumerate(points):
            if point.speed_knots > movement_threshold_knots:
                start_idx = i
                break
        # find the last moving points
        end_idx = len(points) - 1
        for i in range(len(points) - 1, -1, -1):
            if points[i].speed_knots > movement_threshold_knots:
                end_idx = i
                break
        trimmed = points[start_idx:end_idx + 1]
        print(f"Trimmed {start_idx} stationary points from start and {len(points) - 1 - end_idx} from end.")
        return trimmed, start_idx, end_idx

    def simplify_straight_segments(self, points: List[GPSPoint], 
        angle_threshold_degrees=5, min_distance=10) -> List[GPSPoint]:
        """ Remove redundant points along paths that are straight lines"""
        if len(points) < 3:
            return points
        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            prev = simplified[-1]
            curr = points[i]
            next_pt = points[i + 1]

            # calculate breaing from prev to curr and curr to next
            bearing1 = self._calculate_bearing(prev, curr)
            bearing2 = self._calculate_bearing(curr, next_pt)

            # calculate the angle difference
            angle_diff = abs(bearing2 - bearing1)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            # keep point if direction changes very significantly
            if angle_diff > angle_threshold_degrees:
                simplified.append(curr)
            elif prev.distance_to(curr) >= min_distance:
                simplified.append(curr)

        simplified.append(points[-1])
        print(f"Simplified from {len(points)} to {len(simplified)} points.")
        return simplified

    def _calculate_bearing(self, point1: GPSPoint, point2: GPSPoint) -> float:
        """ calculate bearing from point 1 to point 2 in degrees"""
        lat1 = math.radians(point1.latitude)
        lat2 = math.radians(point2.latitude)
        delta_lon = math.radians(point2.longitude - point1.longitude)
        y = math.sin(delta_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def clean_data(self) -> List[GPSPoint]:
        """ performs all of the data cleaning steps"""
        print("\nStarting GPS data cleaning...")
        # removing duplicates
        cleaned = self.remove_duplicates()
        # removing outliers
        cleaned, _, _ = self.remove_outliers(cleaned)
        # trimming stationary endpoints
        cleaned, _, _ = self.trim_stationary_endpoints(cleaned)
        # simplifying straight segments
        cleaned = self.simplify_straight_segments(cleaned)
        print("GPS data cleaning completed.\n")
        return cleaned

        