import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt


"""
3d visualisation module. Contains definition of the transformations done to 'boxes' to show them in 3d.

Functions:
|-  define_faces() - 
|-  plot_face() -
|-  plot_box() -
|-  visualise() -

V.1.01 MS 18/09/23

"""
cmap = matplotlib.colormaps['jet']



def define_faces(box: object) -> list:
    "Moved to separate function for clarity"
    faces = []

    # top face
    y = np.array([box.bl_corner[1], box.br_corner[1], box.tr_corner[1], box.tl_corner[1]])
    x = np.array([box.bl_corner[0], box.br_corner[0], box.tr_corner[0], box.tl_corner[0]]) 
    z = np.array([box.cloud_top, box.cloud_top, box.cloud_top, box.cloud_top])

    top_face = [list(zip(x,y,z))]
    faces.append(top_face)
    

    # bottom face
    y = np.array([box.bl_corner[1], box.br_corner[1], box.tr_corner[1], box.tl_corner[1]])
    x = np.array([box.bl_corner[0], box.br_corner[0], box.tr_corner[0], box.tl_corner[0]])  
    z = np.array([box.cloud_bottom, box.cloud_bottom, box.cloud_bottom, box.cloud_bottom])

    bottom_face = [list(zip(x,y,z))] 
    faces.append(bottom_face)


    # North face
    y = np.array([box.tl_corner[1], box.tr_corner[1], box.tr_corner[1], box.tl_corner[1]])
    x = np.array([box.tl_corner[0], box.tr_corner[0], box.tr_corner[0], box.tl_corner[0]]) 
    z = np.array([box.cloud_bottom, box.cloud_bottom, box.cloud_top, box.cloud_top])

    top_face = [list(zip(x,y,z))]
    faces.append(top_face)


    # South face
    y = np.array([box.bl_corner[1], box.br_corner[1], box.br_corner[1], box.bl_corner[1]])
    x = np.array([box.bl_corner[0], box.br_corner[0], box.br_corner[0], box.bl_corner[0]]) 
    z = np.array([box.cloud_bottom, box.cloud_bottom, box.cloud_top, box.cloud_top])

    top_face = [list(zip(x,y,z))]
    faces.append(top_face)
    

    # East face
    y = np.array([box.bl_corner[1], box.tl_corner[1], box.tl_corner[1], box.bl_corner[1]])
    x = np.array([box.bl_corner[0], box.tl_corner[0], box.tl_corner[0], box.bl_corner[0]]) 
    z = np.array([box.cloud_bottom, box.cloud_bottom, box.cloud_top, box.cloud_top])

    top_face = [list(zip(x,y,z))]
    faces.append(top_face)

    
    # East face
    y = np.array([box.br_corner[1], box.tr_corner[1], box.tr_corner[1], box.br_corner[1]])
    x = np.array([box.br_corner[0], box.tr_corner[0], box.tr_corner[0], box.br_corner[0]]) 
    z = np.array([box.cloud_bottom, box.cloud_bottom, box.cloud_top, box.cloud_top])

    top_face = [list(zip(x,y,z))]
    faces.append(top_face)

    return faces


def plot_face(ax: object, box: object, face: list, edge_color: str, linewidth: float):
    
    plotted_face = Poly3DCollection(face, alpha = 0.5, linewidth=linewidth)
    plotted_face.set_facecolor(cmap(box.cloud_type / 20))
    plotted_face.set_edgecolor(edge_color)
    ax.add_collection3d(plotted_face)


def plot_box(ax: object, box: object, sun_vector: list, output=None):
    "Plot the box by plotting diagrams of all the faces of it"
    
    var = None
    # assert output not in ['index', 'c_type', 'c_top'], \
    #     'Incorrect return type, has to be: index, c_type, c_top'

    if box.sunpath_intersects_box_3d(sun_vector) and sun_vector[1][2] > 0: 
        
        linewidth = 3
        edge_color = 'r'  # Mark passed clouds
        if output == 'index': var = (box.x_id, box.y_id)
        if output == 'c_type': var = box.cloud_type
        if output == 'c_top': var = box.cloud_top

    else: 
        edge_color = 'k'
        linewidth = 1
    
    faces = define_faces(box)
    for f in faces: plot_face(ax, box, f, edge_color, linewidth)

    if var is not None: return var 
    return


def visualise(boxes: list, sun_vector:list, figsize=(8, 8), output=None):

    lab_position = sun_vector[0]
    sun_position = sun_vector[1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d" )

    # Visuals
    ax.set_ylabel('Lat [deg]')
    ax.set_xlabel('Lon [deg]')
    ax.set_zlabel('Height [m]')
    if len(boxes) == 49:
        ax.set_ylim(45.32, 45.73)
        ax.set_xlim(8.95, 9.35)
    else:
        ax.set_ylim(45.15, 45.91)
        ax.set_xlim(8.75, 9.55)
    ax.set_zlim(0, 10_000)
    # plt.grid(alpha=0.5)

    var = []
    colors = []

    for i, b in enumerate(boxes):

        if b.cloud_top: 
            plot = plot_box(ax, b, sun_vector, output)
            if plot: var.append(plot)   # append the output variable if the plot box is marked
        


    ax.scatter(lab_position[0], lab_position[1], lab_position[2], marker='x', s=50, c='r')
    ax.text(x=9.134, y=45.51, z=0, s='Polimi')

    if sun_position[2] > 0: # Show sun vector in red if its above horizon
        ax.plot(
            [lab_position[0], sun_position[0]], 
            [lab_position[1], sun_position[1]],
            [lab_position[2], sun_position[2]],
            linestyle='--', c='r')

    else:                   # Or in grey if not
        ax.plot(
            [lab_position[0], sun_position[0]], 
            [lab_position[1], sun_position[1]],
            [0, 0],
            linestyle='--', c='#9e9796')

    plt.show()

    return var
