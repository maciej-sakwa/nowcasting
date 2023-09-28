import pandas as pd
import numpy as np
import math

"""
Box class module. Contains definition of the cloud 'boxes' and necessary orientation tests to determine the location of the box of the sun path.
V.1.0 MS 18/09/23

Class Box:
    Methods:
    |-  __init__()
    |-  specify_cloud_type()
    |-  sunpath_intersects_box()
    |-  sunpath_intersects_box_3d()

    Private methods:
    |-  __orientation()
    |-  __do_interact()
    |-  __elevation_angle_corner()
    |-  __check_corners()


"""


class Box():

    def __init__(self, flat_id, x_id, y_id, bl_corner, br_corner, tl_corner, tr_corner):
        self.flat_id = flat_id
        self.x_id = x_id
        self.y_id = y_id
        self.bl_corner = bl_corner
        self.br_corner = br_corner
        self.tl_corner = tl_corner
        self.tr_corner = tr_corner

        # Cloud type definition
        self.cloud_bottom = None 
        self.cloud_top = None
        self.cloud_type = None
        

    def specify_cloud_type(self, df_hour: pd.DataFrame, height_list: list):
        """Check cloud type and height. The dataframe has to be indexed with id aligned with box_id"""
        
        row_box = df_hour[df_hour.box_id == self.flat_id].copy()
        c_height = row_box.saf_htop.iloc[0]
        c_type = row_box.saf_ct.iloc[0]

        if c_height < 0:       # If the h_value is equal to -9999 there is no cloud, we return without redefinig the type
            return

        self.cloud_top = c_height 
        self.cloud_type = c_type
        try:
            self.cloud_bottom = height_list[c_type]
        except IndexError:
            self.cloud_bottom = None #TODO think of how to define the bottom if the cloud is not specified (is it necessary in the first place)

        try:
            assert self.cloud_top > self.cloud_bottom, f"Top should be higher than bottom: c_top: {self.cloud_top}, c_bottom: {self.cloud_bottom}, c_type: {self.cloud_type}"
        except TypeError:
            self.cloud_top = None
            self.cloud_bottom = None
        except AssertionError: # Due to data inconsistency cloud types are quite often miss-labelled
            self.cloud_bottom = 1_000
        # if self.cloud_top < self.cloud_bottom: 
        #     print(f'Ctop: {self.cloud_top}, cbottom: {self.cloud_bottom}')

        return 


    def __orientation(self, p, q, r):
        """Determine the orientation of an ordered triplet of points."""
        
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # colinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise

    def __do_interact(self, p1, q1, p2, q2):
        """Check if line segments p1q1 and p2q2 intersect."""
        """The method uses a basic orientation test to determine if the two lines are intersecting"""

        o1 = self.__orientation(p1, q1, p2)
        o2 = self.__orientation(p1, q1, q2)
        o3 = self.__orientation(p2, q2, p1)
        o4 = self.__orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases for colinear points
        return False

    def sunpath_intersects_box(self, vector: list) -> bool:
        """Check if the sun vector intersects with the box."""

        # Check each side of the box
        if self.__do_interact(vector[0], vector[1], self.bl_corner, self.br_corner): return True
        if self.__do_interact(vector[0], vector[1], self.br_corner, self.tr_corner): return True
        if self.__do_interact(vector[0], vector[1], self.tr_corner, self.tl_corner): return True
        if self.__do_interact(vector[0], vector[1], self.tl_corner, self.bl_corner): return True

        return False

    def __elevation_angle_corner(self, corner: list, lab_position: list) -> float:
        
        # corner = [b.br_corner[0], b.br_corner[1], 4000 / 90_000]
        xy_distance = np.linalg.norm(np.array(corner[:2]) - np.array(lab_position[:2]))
        
        return (math.atan(corner[2]/ 90_000 / xy_distance) / np.pi) * 180
        

    def __check_corners(self, sun_vector: list) -> bool:

        angles = []

        corners = [
            [self.bl_corner[0], self.bl_corner[1], self.cloud_bottom],
            [self.br_corner[0], self.br_corner[1], self.cloud_bottom],
            [self.tl_corner[0], self.tl_corner[1], self.cloud_bottom],
            [self.tr_corner[0], self.tr_corner[1], self.cloud_bottom],
            [self.bl_corner[0], self.bl_corner[1], self.cloud_top],
            [self.br_corner[0], self.br_corner[1], self.cloud_top],
            [self.tl_corner[0], self.tl_corner[1], self.cloud_top],
            [self.tr_corner[0], self.tr_corner[1], self.cloud_top],
        ]

        for c in corners: 
            angles.append(self.__elevation_angle_corner(c, sun_vector[0]))

        if min(np.array(angles)) <= self.__elevation_angle_corner(sun_vector[1], sun_vector[0]) <= max(np.array(angles)):
            return True
        
        return False
    
    def sunpath_intersects_box_3d(self, vector: list) -> bool:
        "The sunpath has to intersect with 2 walls of the box, at least"

        flat_vector = [item[:2] for item in vector]

        # Check line segment z-coordinates against box top and bottom
        if not (vector[0][2] > self.cloud_top or vector[1][2] < self.cloud_bottom):

            # Check each side (rectangle) of the box for intersection
            if self.sunpath_intersects_box(flat_vector) and self.__check_corners(vector): return True

        return False
