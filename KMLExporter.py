import pandas as pd
from typing import Optional


class KMLExporter:
    """decorate a KML file based on the GPS analyzer and export"""

    def __init__(self, df: pd.DataFrame, stops_df: pd.DataFrame = None,
                 left_turns_df: pd.DataFrame = None):
        """
        init KML exporter
        Args:
            df: Main GPS data DataFrame
            stops_df: DataFrame with stop information
            left_turns_df: DataFrame with left turn information
        """
        self.df = df
        # error checks for if data has no stops or left turns
        self.stops_df = stops_df if stops_df is not None else pd.DataFrame()
        self.left_turns_df = left_turns_df if left_turns_df is not None else pd.DataFrame()

    def generate_kml(self, output_filename: str, trip_name: str = "Path"):
        """
        create the KML file and save to current dir

        Args:
            output_filename: Path to output KML file
            trip_name: Name for the trip
        """
        kml_content = []

        # KML Header
        kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml_content.append('<Document>')
        kml_content.append(f'  <name>{trip_name}</name>')
        kml_content.append(f'  <description>GPS track with {len(self.df)} points</description>')

        # Add styles
        self.add_styles(kml_content)

        # Add route path(s)
        self.add_route_path(kml_content)

        # Add markers
        self.start_end_markers(kml_content)
        self.stop_markers(kml_content)
        self.turn_markers(kml_content)

        # KML Footer
        kml_content.append('</Document>')
        kml_content.append('</kml>')

        # Write to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kml_content))

    def add_styles(self, kml_content: list):

        # add the route path to file
        kml_content.extend([
            '  <Style id="routeStyle">',
            '    <LineStyle>',
            '      <color>ff00ffff</color>',  # use AABBGGRR format for yellow
            '      <width>4</width>',
            '    </LineStyle>',
            '  </Style>',
        ])

        # add stops to file
        kml_content.extend([
            '  <Style id="stopStyle">',
            '    <IconStyle>',
            '      <color>ff0000ff</color>',  # red
            '      <scale>1.3</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>',  # google maps icon
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # add left turns
        kml_content.extend([
            '  <Style id="leftTurnStyle">',
            '    <IconStyle>',
            '      <color>ff00ffff</color>',  # Yellow
            '      <scale>1.1</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png</href>',  # google maps icon
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # add stop to the file
        kml_content.extend([
            '  <Style id="startStyle">',
            '    <IconStyle>',
            '      <color>ff00ff00</color>',  # Green
            '      <scale>1.5</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/go.png</href>', # google maps icon
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

        # add ends to the file
        kml_content.extend([
            '  <Style id="endStyle">',
            '    <IconStyle>',
            '      <color>ff0000ff</color>',
            '      <scale>1.5</scale>',
            '      <Icon>',
            '        <href>http://maps.google.com/mapfiles/kml/paddle/stop.png</href>',  # google maps icon
            '      </Icon>',
            '    </IconStyle>',
            '  </Style>',
        ])

    def add_route_path(self, kml_content: list):

        # start route line
        kml_content.extend([
            '  <Placemark>',
            '    <name>Route</name>',
            '    <styleUrl>#routeStyle</styleUrl>',
            '    <LineString>',
            '      <tessellate>1</tessellate>',
            '      <altitudeMode>clampToGround</altitudeMode>',
            '      <coordinates>',
        ])

        for idx, coord in self.df.iterrows():
            altitude = 3  # we don't care about alt. so hardcode to 3
            kml_content.append(
                f'        {coord["longitude"]:.6f},{coord["latitude"]:.6f},{altitude:.1f}'

            )

        # close out the points kml tags
        kml_content.extend([
            '      </coordinates>',
            '    </LineString>',
            '  </Placemark>',
        ])

    def start_end_markers(self, kml_content: list):

        # end if df is empty
        if len(self.df) == 0:
            return

        # create start at first idx of df
        start = self.df.iloc[0]
        # add it to the kml file
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

        # create end at last idx of df
        end = self.df.iloc[-1]
        # add to kml file
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

    def stop_markers(self, kml_content: list):

        # if car never stops, return
        if len(self.stops_df) == 0:
            return

        # iterate through all stops in the dataframe
        for idx, stop in self.stops_df.iterrows():
            # get the duration of the stop from
            duration_str = f"{stop['duration']:.1f}s"
            # if stop is significant
            if stop['duration'] >= 60:
                # truncate duration for plotting (so we don't plot multiple stops)
                duration_str = f"{stop['duration'] / 60:.1f}m"

            # append to the kml file rows
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

    def turn_markers(self, kml_content: list):

        # mark left turns as yellow
        # if there are turns detected in the df
        if len(self.left_turns_df) > 0:
            # iterate through them
            for idx, turn in self.left_turns_df.iterrows():
                # get the angle of direction
                angle = abs(turn.get('heading_change', turn.get('bearing_change', 0)))

                # add to KML file
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
