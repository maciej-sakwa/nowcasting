import math
import numpy as np 
import pandas as pd

from pvlib import solarposition

from src_cloud_tracking.boxes import Box
from src_cloud_tracking import box_definition_utils
from src_cloud_tracking import visualisation_3d
from src_cloud_tracking import visualisation_2d

############################################# PARAMETERS #############################################

N_PIXELS = 15
LAB_POSITION = [9.15, 45.5, 0]
HOUR = '2023-08-26 17:25:00'
SUN_DISTANCE = 1
SUN_HEIGHT = 90_000

# Cloud height definition - according to **

# 0	    non-processed containing no data or corrupt	
# 1	    cloud free land no contamination by snow/ic	
# 2	    cloud free sea no contamination by snow/ice	
# 3	    land contaminated by snow	
# 4	    sea contaminated by snow/ice	
# 5	    very low and cumuliform clouds  	
# 6	    very low and stratiform clouds	
# 7	    low and cumuliform clouds  	
# 8	    low and stratiform clouds	
# 9	    medium and cumuliform clouds	
# 10	medium and stratiform clouds	
# 11	high opaque and cumuliform clouds	
# 12	high opaque and stratiform clouds	
# 13	very high opaque and cumuliform clouds	
# 14	very high opaque and stratiform clouds	
# 15	high semi-transparent thin clouds 	
# 16	high semi-transparent meanly thick clouds	
# 17	high semi-transparent thick clouds 	
# 18	high semi-transparent above low or medium cloud	
# 19	fractional clouds	
# 20	undefined (undefined by CMa)	

CLOUD_MIN = [
    None, None, None, None, None, 0, 0, 0, 0, 2_000, 2_000, 2_000, 2_000, 6_096, 6_096, 6_096, 6_096, 6_096, 6_096, 500, None
    ]
assert len(CLOUD_MIN) == 21, 'Specify the cloud bottom for all 21 cloud types'


############################################# PARAMETERS #############################################


def get_sun_position(time: str, lab_pos: list, sun_dist: int, height: int) -> list:
    
    solar_angles = solarposition.get_solarposition(time, longitude = lab_pos[0], latitude=lab_pos[1])
    azimuth_angle = solar_angles.azimuth.iloc[0]
    elevation_angle = solar_angles.elevation.iloc[0]

    print(elevation_angle)

    x_diff = sun_dist * math.sin((180 - azimuth_angle) / 180 * math.pi)
    y_diff = sun_dist * math.cos((azimuth_angle) / 180 * math.pi)
    z_diff = sun_dist * math.tan((elevation_angle) / 180 * math.pi) * height

    return [LAB_POSITION[0] + x_diff, LAB_POSITION[1] + y_diff, LAB_POSITION[2] + z_diff]



def main():
    
    # Load data from the stored file, select the hour of interest, and convert the cloud type to int
    # df_satellite = pd.read_parquet(path = r'dataset/dataframes/sat_data_full.parquet.gzip')
    df_satellite = pd.read_parquet(path = r'dataset/dataframes/new_sat_data_full.parquet.gzip')
    df_satellite_hour = df_satellite[df_satellite.data_val == HOUR].copy()
    df_satellite_hour.saf_ct = df_satellite_hour.saf_ct.astype(int)

    # Define the sun location for the hour of interest, define the sun vector connecting sun and lab location
    sun_location = get_sun_position(HOUR +'+0000', LAB_POSITION, SUN_DISTANCE, SUN_HEIGHT)
    sun_vector = [
        [LAB_POSITION[0], LAB_POSITION[1], LAB_POSITION[2]],
        [sun_location[0], sun_location[1], sun_location[2]]
        ]

    # Define the boxes
    boxes = box_definition_utils.set_up_boxes(Box, df_satellite_hour, N_PIXELS)

    # Align indices
    df_satellite_hour['box_id'] = df_satellite_hour.apply(
        lambda row: box_definition_utils.find_x_id_big(row)*N_PIXELS + box_definition_utils.find_y_id_big(row), axis=1)

    # Update boxes with data from the loaded sattelite df according to their index. 
    for i, b in enumerate(boxes):
        b.specify_cloud_type(df_satellite_hour, CLOUD_MIN)


    # Tests and vis
    visualisation_2d.visualise(boxes, sun_vector)
    visualisation_3d.visualise(boxes, sun_vector)



if __name__ == '__main__':
    main()