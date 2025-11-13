import pandas as pd
from typing import Optional


class KMLExporter:
    """Export GPS data to KML format with markers for stops and turns"""

    def __init__(self, df: pd.DataFrame, stops_df: pd.DataFrame = None,
                 left_turns_df: pd.DataFrame = None, right_turns_df: pd.DataFrame = None):
        """
        Initialize KML exporter

        Args:
            df: Main GPS data DataFrame
            stops_df: DataFrame with stop information
            left_turns_df: DataFrame with left turn information
            right_turns_df: DataFrame with right turn information (optional)
        """
        self.df = df
        self.stops_df = stops_df if stops_df is not None else pd.DataFrame()
        self.left_turns_df = left_turns_df if left_turns_df is not None else pd.DataFrame()
        self.right_turns_df = right_turns_df if right_turns_df is not None else pd.DataFrame()

    def generate_kml(self, output_filename: str, trip_name: str = "GPS Track",
                     max_points_per_path: int = 10000):
        """
        Generate KML file with route line and markers

        Args:
            output_filename: Path to output KML file
            trip_name: Name for the trip
            max_points_per_path: Maximum points per path (KML limitation)
        """
        print(f"\n=== Generating KML: {output_filename} ===")

        kml_content = []

        # KML Header
        kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml_content.append('<Document>')
        kml_content.append(f'  <name>{trip_name}</name>')
        kml_content.append(f'  <description>GPS track with {len(self.df)} points</description>')

        # Add styles
        self._add_styles(kml_content)

        # Add route path(s)
        self._add_route_paths(kml_content, max_points_per_path)

        # Add markers
        self._add_start_end_markers(kml_content)
        self._add_stop_markers(kml_content)
        self._add_turn_markers(kml_content)

        # KML Footer
        kml_content.append('</Document>')
        kml_content.append('</kml>')

        # Write to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kml_content))

        print(f"✓ KML file created successfully")
        print(f"  - Route points: {len(self.df)}")
        print(f"  - Stops: {len(self.stops_df)}")
        print(f"  - Left turns: {len(self.left_turns_df)}")
        if len(self.right_turns_df) > 0:
            print(f"  - Right turns: {len(self.right_turns_df)}")
        print(f"\nOpen {output_filename} in Google Earth to view")

    def _add_styles(self, kml_content: list):
        """Add KML style definitions"""

        # Yellow route line
        kml_content.extend([
            '  <Style id="routeStyle">',
            '    <LineStyle>',
            '      <color>ff00ffff</color>',  # AABBGGRR format: yellow
            '      <width>4</width>',
            '    </LineStyle>',
            '  </Style>',
        ])

        # Red stop markers
        kml_content.extend([
            '  <Style id="stopStyle">',
            '    <IconStyle>',
            '      <color>ff0000ff</color>',  # Red
            '      <scale>1.3</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>',
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # Yellow left turn markers
        kml_content.extend([
            '  <Style id="leftTurnStyle">',
            '    <IconStyle>',
            '      <color>ff00ffff</color>',  # Yellow
            '      <scale>1.1</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png</href>',
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # Green right turn markers (optional)
        kml_content.extend([
            '  <Style id="rightTurnStyle">',
            '    <IconStyle>',
            '      <color>ff00ff00</color>',  # Green
            '      <scale>1.1</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/grn-blank.png</href>',
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # Start marker (green flag)
        kml_content.extend([
            '  <Style id="startStyle">',
            '    <IconStyle>',
            '      <color>ff00ff00</color>',
            '      <scale>1.5</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/go.png</href>',
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # End marker (red flag)
        kml_content.extend([
            '  <Style id="endStyle">',
            '    <IconStyle>',
            '      <color>ff0000ff</color>',
            '      <scale>1.5</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/stop.png</href>',
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

    def _add_route_paths(self, kml_content: list, max_points_per_path: int):
        """Add route line(s) - split if too many points"""

        total_points = len(self.df)
        num_paths = (total_points // max_points_per_path) + 1

        if num_paths > 1:
            print(f"  Splitting route into {num_paths} paths (max {max_points_per_path} points each)")

        for path_num in range(num_paths):
            start_idx = path_num * max_points_per_path
            end_idx = min((path_num + 1) * max_points_per_path, total_points)

            path_df = self.df.iloc[start_idx:end_idx]

            kml_content.extend([
                '  <Placemark>',
                f'    <name>Route {path_num + 1}</name>' if num_paths > 1 else '    <name>Route</name>',
                '    <styleUrl>#routeStyle</styleUrl>',
                '    <LineString>',
                '      <tessellate>1</tessellate>',
                '      <altitudeMode>clampToGround</altitudeMode>',
                '      <coordinates>',
            ])

            # Add coordinates (lon,lat,alt - note: KML uses lon,lat order!)
            for _, point in path_df.iterrows():
                # Use altitude if available, otherwise use 3m above ground
                alt = point.get('altitude', 3)
                kml_content.append(
                    f'        {point["longitude"]:.6f},{point["latitude"]:.6f},{alt:.1f}'
                )

            kml_content.extend([
                '      </coordinates>',
                '    </LineString>',
                '  </Placemark>',
            ])

    def _add_start_end_markers(self, kml_content: list):
        """Add start and end markers"""

        if len(self.df) == 0:
            return

        # Start marker
        start = self.df.iloc[0]
        kml_content.extend([
            '  <Placemark>',
            '    <name>Start</name>',
            f'    <description>Start time: {start["timestamp"]}</description>',
            '    <styleUrl>#startStyle</styleUrl>',
            '    <Point>',
            f'      <coordinates>{start["longitude"]:.6f},{start["latitude"]:.6f},3</coordinates>',
            '    </Point>',
            '  </Placemark>',
        ])

        # End marker
        end = self.df.iloc[-1]
        kml_content.extend([
            '  <Placemark>',
            '    <name>End</name>',
            f'    <description>End time: {end["timestamp"]}</description>',
            '    <styleUrl>#endStyle</styleUrl>',
            '    <Point>',
            f'      <coordinates>{end["longitude"]:.6f},{end["latitude"]:.6f},3</coordinates>',
            '    </Point>',
            '  </Placemark>',
        ])

    def _add_stop_markers(self, kml_content: list):
        """Add red markers for detected stops"""

        if len(self.stops_df) == 0:
            return

        for idx, stop in self.stops_df.iterrows():
            duration_str = f"{stop['duration']:.1f}s"
            if stop['duration'] >= 60:
                duration_str = f"{stop['duration'] / 60:.1f}m"

            kml_content.extend([
                '  <Placemark>',
                f'    <name>Stop {idx + 1}</name>',
                f'    <description>Duration: {duration_str}</description>',
                '    <styleUrl>#stopStyle</styleUrl>',
                '    <Point>',
                f'      <coordinates>{stop["longitude"]:.6f},{stop["latitude"]:.6f},3</coordinates>',
                '    </Point>',
                '  </Placemark>',
            ])

    def _add_turn_markers(self, kml_content: list):
        """Add markers for detected turns"""

        # Left turns (yellow)
        if len(self.left_turns_df) > 0:
            for idx, turn in self.left_turns_df.iterrows():
                angle = abs(turn.get('heading_change', turn.get('bearing_change', 0)))

                kml_content.extend([
                    '  <Placemark>',
                    f'    <name>Left Turn {idx + 1}</name>',
                    f'    <description>Angle: {angle:.1f}°</description>',
                    '    <styleUrl>#leftTurnStyle</styleUrl>',
                    '    <Point>',
                    f'      <coordinates>{turn["longitude"]:.6f},{turn["latitude"]:.6f},3</coordinates>',
                    '    </Point>',
                    '  </Placemark>',
                ])

        # Right turns (green) - optional
        if len(self.right_turns_df) > 0:
            for idx, turn in self.right_turns_df.iterrows():
                angle = abs(turn.get('heading_change', turn.get('bearing_change', 0)))

                kml_content.extend([
                    '  <Placemark>',
                    f'    <name>Right Turn {idx + 1}</name>',
                    f'    <description>Angle: {angle:.1f}°</description>',
                    '    <styleUrl>#rightTurnStyle</styleUrl>',
                    '    <Point>',
                    f'      <coordinates>{turn["longitude"]:.6f},{turn["latitude"]:.6f},3</coordinates>',
                    '    </Point>',
                    '  </Placemark>',
                ])

    def generate_simplified_kml(self, output_filename: str,
                                simplification_factor: int = 10,
                                trip_name: str = "GPS Track (Simplified)"):
        """
        Generate a simplified KML with fewer points (for large datasets)

        Args:
            output_filename: Path to output file
            simplification_factor: Keep every Nth point
            trip_name: Name for the trip
        """
        print(f"\n=== Generating Simplified KML ===")
        print(f"  Simplification: 1 in {simplification_factor} points")

        # Simplify dataframe but keep stops and turns
        simplified_df = self.df.iloc[::simplification_factor].copy()

        # Create temporary exporter with simplified data
        temp_exporter = KMLExporter(
            simplified_df,
            self.stops_df,
            self.left_turns_df,
            self.right_turns_df
        )

        temp_exporter.generate_kml(output_filename, trip_name)