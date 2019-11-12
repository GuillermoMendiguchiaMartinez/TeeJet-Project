import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyproj as proj
from tqdm import tqdm


def global2local2(point, ref_point):
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic coordinate system

    # Erect own local flat cartesian coordinate system
    cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(ref_point[1], ref_point[0]))
    x, y = proj.transform(crs_wgs, cust, point[0], point[1])
    return x,-y                     # Multiply by a minus because the latitude is higher the more "up" you go


def local2global(point, ref_point):
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic coordinate system

    # Erect own local flat cartesian coordinate system
    cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(ref_point[1], ref_point[0]))
    x, y = proj.transform(cust, crs_wgs, point[0], point[1])
    return x, y

def global2local(point, ref_point):
    c_earth = 40075000
    print('Point 0 = ', point[0])
    print('Ref_Point 0 = ', ref_point[0])
    print('Point 1 = ', point[1])
    print('Ref_Point 1 = ', ref_point[1])
    print('Point 1 - Ref_Point 1 = ', point[1] - ref_point[1])

    dlat = ref_point[1] - point[1]
    dlon = point[0] - ref_point[0]

    dx = dlon * c_earth * math.cos((ref_point[1] + point[1]) * math.pi / 360) / 360
    dy = dlat * c_earth / 360
    return dx, dy


def waypoints(coordinates,cam_fov_deg, height, overlap,pixel_size=0.5):
    '''
    :param coordinates: list of global coordinates for the map, corners of the area.
     List of tuples of lat long. [(lat_1,lon_1),...(lat_n,lon_n)]
    :param cam_fov_deg: Field of view of the camera
    :param height: Height
    :param overlap:
    :param pixel_size
    :return: waypoints Tuple of lat long [(lat_1,lon_1),...(lat_n,lon_n)] in global form.
    '''

    cam_fov = math.radians(cam_fov_deg / 2)

    # Initialisation of latitude longitude
    lat_max = coordinates[0][1]
    lat_min = coordinates[0][1]
    lon_max = coordinates[0][0]
    lon_min = coordinates[0][0]

    for coordinate in coordinates:
        if coordinate[0] > lon_max:
            lon_max = coordinate[0]

        elif coordinate[0] < lon_min:
            lon_min = coordinate[0]

        if coordinate[1] > lat_max:
            lat_max = coordinate[1]

        elif coordinate[1] < lat_min:
            lat_min = coordinate[1]

    dlat = lat_max - lat_min
    dlon = lon_max - lon_min

    Global2local = global2local2((lon_max, lat_min), (lon_min, lat_max))
    dx = Global2local[0]/pixel_size
    dy = Global2local[1]/pixel_size
    print('Global2Local = ', Global2local)

    # Create grid cell sizes with overlap 0.5
    cell_size_x = round(int((math.tan(cam_fov) * height * overlap)/pixel_size))
    cell_size_y = round(int((math.tan(cam_fov) * height * overlap)/pixel_size))

    # Number of cells in the grid
    n_grids_x = np.ceil(dx / cell_size_x)
    n_grids_y = np.ceil(dy / cell_size_y)

    print('Number of grids in X = ', n_grids_x)
    print('Number of grids in Y = ', n_grids_y)

    n_grids_total = n_grids_x * n_grids_y
    print('Total number of grids = ', n_grids_total)

    # Create a map of zeros
    map_array = np.zeros((int(cell_size_y*n_grids_y), int(cell_size_x*n_grids_x)))

    coordinates_rel = []
    ref_point = (lon_min, lat_max)          # We start at top left

    # Create the local coordinates vector for the real coordinates gotten from GUI.
    for coordinate in coordinates:
        coordinates_rel.append(np.ceil(np.array(global2local2(coordinate, ref_point))/pixel_size).astype('int'))

    # Transform to array (N by 2)
    coord_rel_np = np.array(coordinates_rel)

    pts = coord_rel_np.reshape((-1, 1, 2))
    #print(pts)

    cv2.fillPoly(map_array, [pts], 1)
#    plt.imshow(map_array, cmap='gray')
#    plt.show()

    ''' Return center points of grid'''
    count_ones=0

    centers = []
    for j in range(int(n_grids_x)):
        for i in range(int(n_grids_y)):
            patch_l = map_array[cell_size_y*i:cell_size_y*i+cell_size_y, cell_size_x*j:cell_size_x*j+cell_size_x]
            #print(patch_l)
            if patch_l.max() != 0:
                count_ones += np.sum(patch_l)

                # calculate center of patch. Should be length of lon + half of a patch size
                x_center = cell_size_x*j+cell_size_x*0.5
                y_center = cell_size_x*i+cell_size_x*0.5
                coords = (x_center-0.5, y_center-0.5)
                centers.append(coords)
    print("Local coordinates (x,y)")
    print(centers)
    map_color=cv2.cvtColor((map_array*255).astype('uint8'),cv2.COLOR_GRAY2BGR)
    for center in centers:
        map_color[int(center[1]),int(center[0])] = (255,0,0)
    #plt.imshow(map_color)
    #plt.show()
    waypts_m=[]
    for center in centers:
        waypts_m.append((center[0]*pixel_size, center[1]*pixel_size))
    lonlat_wpts = []
    for waypt in tqdm(waypts_m):
        lonlat_wpts.append(local2global(waypt, ref_point))

    print("Global coordinates are (lat, lon):", lonlat_wpts)

if __name__ == '__main__':
    corners = [(9.93758976, 57.04503142),(9.93810475,57.04265598),(9.94246066,57.04413262)]
    waypoints(corners, 90, 25, 0.5, 0.5)

