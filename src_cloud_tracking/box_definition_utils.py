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

def find_y_id(row: pd.Series):
    "Specify the y or lat-wise index"

    y_id = int(str(row.id_msg)[2])
    
    return  np.abs(9 - y_id)


def find_x_id(row: pd.Series):
    "Specify the y or lat-wise index"

    x_id = int(str(row.id_msg)[-2:])

    return np.abs(x_id - 47)


def set_up_boxes(box: object, data: pd.DataFrame, n_pixels: int):

    boxes = []
    full_mesh = []

    # Initialize mesh like structure of the data - the transformation has the purpose of setting up the array in the correct 'visual' order
    y_data = data.lat.values
    y_mesh = np.array([y_data[0+7*i:7+7*i] for i in range(7)])
    x_data = data.lon.values
    x_mesh = np.array([x_data[0+7*i:7+7*i] for i in range(7)])

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

