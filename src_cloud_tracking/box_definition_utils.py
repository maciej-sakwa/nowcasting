import numpy as np
import pandas as pd

"""
Box functions module. Contains definition of the transformations done to 'boxes'.
V.1.0 MS 18/09/23

Functions:
|-  find_y_id
|-  find_x_id
|-  set_up_boxes

"""



# The satellite pixels have their own id eg. 273547
# The third value (eg. 3 in the example above) specifies the 'row' or lat-wise position. In the data it ranges from 3 to 9.
# The final two digits (eg. 47 in the example above) specity the 'column' or lon-wise position. In the data it ranges from 47 to 53.

# The boxes are numbered from left bottom corner and going up
# The sattelite images are numbered from left top corner and going right

# These functions align the indices of df and boxes

def find_y_id(row: pd.Series) -> int:
    "Specify the y or lat-wise index"

    y_id = int(str(row.id_msg)[2])
    
    return  np.abs(9 - y_id)


def find_x_id(row: pd.Series) -> int:
    """Specify the y or lat-wise index

    Args:
        row (pd.Series): _description_

    Returns:
        int: _description_
    """
    

    x_id = int(str(row.id_msg)[-2:])

    return np.abs(x_id - 47)


def set_up_boxes(box: object, data: pd.DataFrame, n_pixels: int) -> list:
    """_summary_

    Args:
        box (object): _description_
        data (pd.DataFrame): _description_
        n_pixels (int): _description_

    Returns:
        list: _description_
    """

    boxes = []
    full_mesh = []

    # Initialize mesh like structure of the data - the transformation has the purpose of setting up the array in the correct 'visual' order
    y_data = data.lat.values
    y_mesh = np.array([y_data[0+n_pixels*i:n_pixels+n_pixels*i] for i in range(n_pixels)])
    x_data = data.lon.values
    x_mesh = np.array([x_data[0+n_pixels*i:n_pixels+n_pixels*i] for i in range(n_pixels)])

    # Find the starting point (in the bottom left corner)
    start_point = (x_mesh[-1, 0], y_mesh[-1, 0])

    # Identify the mesh grid variation both in x and y - each grid cell is shaped like a parallelogram
    delta_y_small = y_mesh[-1, 1] - y_mesh[-1, 0]
    delta_y_big = y_mesh[-2, 0] - y_mesh[-1, 0]

    delta_x_small = x_mesh[-2, 0] - x_mesh[-1, 0]
    delta_x_big = x_mesh[-1, 1] - x_mesh[-1, 0]


    # Identify the corners of the first box 
    x_start, y_start = start_point

    bl_corner_start = [x_start - 0.5*(delta_x_big + delta_x_small), y_start - 0.5*(delta_y_big + delta_y_small)]
    br_corner_start = [x_start + 0.5*(delta_x_big - delta_x_small), y_start - 0.5*(delta_y_big - delta_y_small)]
    tl_corner_start = [x_start - 0.5*(delta_x_big - delta_x_small), y_start + 0.5*(delta_y_big - delta_y_small)]
    tr_corner_start = [x_start + 0.5*(delta_x_big + delta_x_small), y_start + 0.5*(delta_y_big + delta_y_small)]

    # Loop over other boxes 
    for i in range(n_pixels):
        for j in range(n_pixels):
            
            single_box = box(
                flat_id = i * n_pixels + j,
                x_id = i,
                y_id = j,
                bl_corner = [bl_corner_start[0] + i * delta_x_big + j * delta_x_small, bl_corner_start[1] + j * delta_y_big + i * delta_y_small],
                br_corner = [br_corner_start[0] + i * delta_x_big + j * delta_x_small, br_corner_start[1] + j * delta_y_big + i * delta_y_small],
                tl_corner = [tl_corner_start[0] + i * delta_x_big + j * delta_x_small, tl_corner_start[1] + j * delta_y_big + i * delta_y_small],
                tr_corner = [tr_corner_start[0] + i * delta_x_big + j * delta_x_small, tr_corner_start[1] + j * delta_y_big + i * delta_y_small]
            )

            boxes.append(single_box)

            # full_mesh.append([bl_corner_start[0] + i * delta_x_big + j * delta_x_small, bl_corner_start[1] + j * delta_y_big + i * delta_y_small])

    return boxes



def compile_cloud_dataset(satellite_data: pd.DataFrame or str) -> None:
    pass
'''Create a function out of these code, input -> satellite data or folder path, output -> dictionary of cloud data'''
# unique_dates = np.sort(df_satellite.date_time.unique())
# delta = timedelta(minutes=10)
# cloud_data = {}


# for i, date in tqdm(enumerate(unique_dates)):

#     boxes = []

#     # Select data
#     df_satellite_date = df_satellite[df_satellite.date_time == date].copy()
#     df_satellite_date.saf_ct = df_satellite_date.saf_ct.astype(int)

#     # Convert the datetime format to string
#     date = pd.to_datetime(date) - delta
#     date_string = datetime.strftime(date, '%Y-%m-%d %H:%M')

#     # Find sun location
#     sun_location = get_sun_position(date, LAB_POSITION, SUN_DISTANCE, SUN_HEIGHT)
#     sun_vector = [
#         [LAB_POSITION[0], LAB_POSITION[1], LAB_POSITION[2]],
#         [sun_location[0], sun_location[1], sun_location[2]]
#     ]

#     # Set up boxes
#     boxes = box_definition_utils.set_up_boxes(Box, df_satellite_date, 7)

#     # Align indices
#     df_satellite_date['box_id'] = df_satellite_date.apply(lambda row: box_definition_utils.find_x_id(row)*7 + box_definition_utils.find_y_id(row), axis=1)

#     # Update boxes with data from the loaded sattelite df according to their index. 
#     for i, b in enumerate(boxes):
#         b.specify_cloud_type(df_satellite_date, CLOUD_MIN)


#     # Initialize the output matrices
#     cloud_cover = np.zeros((N_PIXELS, N_PIXELS))
#     cloud_passed = np.zeros((N_PIXELS, N_PIXELS))


#     for b in boxes:
#         # Cloud thickness matrix
#         if b.cloud_top is not None:
#             cloud_cover[b.x_id, b.y_id] = (b.cloud_top - b.cloud_bottom) / 10_000
#             # Passed cloud matrix
#             if b.sunpath_intersects_box_3d(sun_vector): 
#                 cloud_passed[b.x_id, b.y_id] = 1
   
   
#     added_data = np.stack((cloud_cover, cloud_passed), axis = 2)
#     cloud_data[date_string] = added_data

# with open(r'dataset/cloud_data.pickle', 'wb') as file:
#     pickle.dump(cloud_data, file)