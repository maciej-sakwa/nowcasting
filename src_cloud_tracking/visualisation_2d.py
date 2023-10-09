import math
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

"""
Box functions module. Contains definition of the transformations done to 'boxes'.
V.1.0 MS 18/09/23

Functions:
|-  visualise()

"""

def visualise(boxes: list, sun_vector:list):
    "Visualise the thickness of the clouds in a colormap"

    fig, ax = plt.subplots(figsize = (10, 8))

    lab_position = sun_vector[0]
    sun_position = sun_vector[1]

    # Visuals
    plt.ylabel('Lat')
    plt.xlabel('Lon')
    if len(boxes) == 49:
        ax.set_ylim(45.32, 45.73)
        ax.set_xlim(8.95, 9.35)
    else:
        ax.set_ylim(45.15, 45.91)
        ax.set_xlim(8.75, 9.55)
    plt.grid(alpha=0.5)

    patches = []
    colors = []

    for i, b in enumerate(boxes):

        
        if b.sunpath_intersects_box(sun_vector) and sun_position[2] > 0:
            
            y = np.array([
                [b.bl_corner[0], b.bl_corner[1]], 
                [b.br_corner[0], b.br_corner[1]],
                [b.tr_corner[0], b.tr_corner[1]],
                [b.tl_corner[0], b.tl_corner[1]], 
                ])

            marked_cell = Polygon(y, facecolor = 'none', edgecolor='r', alpha = 0.8)
            ax.add_patch(marked_cell)
            
        if b.cloud_top:

            y = np.array([
                [b.bl_corner[0], b.bl_corner[1]], 
                [b.br_corner[0], b.br_corner[1]],
                [b.tr_corner[0], b.tr_corner[1]],
                [b.tl_corner[0], b.tl_corner[1]], 
                ])

            cell = Polygon(y, closed=True)
            patches.append(cell)
            colors.append(b.cloud_top - b.cloud_bottom)    
        
        
        # plt.scatter(b.bl_corner[0], b.bl_corner[1], c='k', s=5)
        # plt.scatter(b.br_corner[0], b.br_corner[1], c='k', s=5)
        # plt.scatter(b.tl_corner[0], b.tl_corner[1], c='k', s=5)
        # plt.scatter(b.tr_corner[0], b.tr_corner[1], c='k', s=5)

    p = PatchCollection(patches, alpha=0.4)
    p.set_array(colors)
    ax.add_collection(p)
    fig.colorbar(p, ax=ax, label='Cloud thickness')

    plt.scatter(x=lab_position[0], y=lab_position[1], marker='x', s=50, c='r')
    plt.text(x=9.134, y=45.51, s='Polimi')

    if sun_position[2] > 0: # Show sun vector in red if its above horizon
        plt.plot(
            [lab_position[0], sun_position[0]], 
            [lab_position[1], sun_position[1]],
            linestyle='--', c='r')
    else:                   # Or in grey if not
        plt.plot(
            [lab_position[0], sun_position[0]], 
            [lab_position[1], sun_position[1]],
            linestyle='--', c='#9e9796')


    plt.show()