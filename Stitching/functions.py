import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import glob
import os
from scipy.sparse import lil_matrix
from tqdm import tqdm
from scipy.optimize import least_squares
import geopy.distance
from GPSPhoto import gpsphoto
import multiprocessing as mp
from scipy.sparse.csgraph import dijkstra
import pyexiv2
from datetime import datetime

import copyreg


def patch_cv2_pickiling():
    """
    makes cv2 keypoints and dmatches pickable so they can be transfered into multiprocessing.pool
    :return:
    """

    # Create the bundling between class and arguments to save for Keypoint class
    # See : https://stackoverflow.com/questions/50337569/pickle-exception-for-cv2-boost-when-using-multiprocessing/50394788#50394788
    def _pickle_keypoint(keypoint):  # : cv2.KeyPoint
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )

    # C++ Constructor, notice order of arguments :
    # KeyPoint (float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

    # Apply the bundling to pickle
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)
    #same repeated for dmatches
    def _pickle_dmatch(dmatch):  # : cv2.DMatch
        return cv2.DMatch, (
            dmatch.queryIdx,
            dmatch.trainIdx,
            dmatch.imgIdx,
            dmatch.distance
        )

    copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)


def mc_matcher(args):
    """
    matcher designed to be called by pool.map
    :param args: input arguments in format (position,target,sift_threshold,use_inliers_only,features,goodmatchs_threshold,ransac_threshold)
    :return: (position, target, matches, M) where M is the homography
    """
    position = args[0]
    target = args[1]
    sift_threshold = args[2]
    use_inliers_only = args[3]
    features = args[4]
    goodmatches_threshold = args[5]
    ransac_threshold = args[6]
    # setting up the matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=75)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # extracting the keypoints
    all_matches = flann.knnMatch(features[position][1], features[target][1], k=2)
    goodmatches = []
    for m, n in all_matches:
        if m.distance < sift_threshold * n.distance:
            goodmatches.append(m)

    print("process %d: matching %d and %d got %d matches" % (os.getpid(), position, target, len(goodmatches)))
    #discrad connections that don't have enough good matches
    if len(goodmatches) < goodmatches_threshold:
        return False

    # matching
    keypoints1, _ = features[position]
    keypoints2, _ = features[target]
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in goodmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in goodmatches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    # generating the output

    if use_inliers_only:
        goodmatches = np.array(goodmatches)
        goodmatches = goodmatches.reshape(mask.shape)
        matches = goodmatches[mask == 1]
    else:
        matches = goodmatches

    print("process %d: kept %d matches" % (os.getpid(), len(matches)))
    patch_cv2_pickiling()
    if len(matches) < 4:
        return False
    return (position, target, matches, M)


def homography_path_iter(position, target, predecessors, homographies_rel, homography=np.eye(3)):
    """
    iterativly calculates the resulting absolute homography by multiplying the relative homographies along the path given by predecessors
    :param position: index of current image
    :param target: index of target image (normally the center_image)
    :param predecessors: list of the predecessors for every image
    :param homographies_rel: relative homographies
    :param homography: homography between starting point and current point, is identity matrix if current=starting point
    :return: absolute homography between current position and target position, if a connection exists, otherwise returns False
    """
    if position == target:
        return homography
    elif predecessors[position] == -9999:
        print("one or more images are not connected to the central image")
        print("central image index is:%d" % (target))
        print("failed to connect image with index: %d" % (position))
        return False
    else:
        homography = np.matmul(homographies_rel[(predecessors[position], position)], homography)

        return homography_path_iter(predecessors[position], target, predecessors, homographies_rel, homography)



def perform_djikstra(Graph, start_index):
    """
    performs the djikstra algorithm on a graph and lets you choose a starting node
    :param Graph: Graph that the algorithm should be performed on
    :param start_index: starting node
    :return: cost: cost to get to each node (-9999 for starting node or nodes that are not accesible)
            predecesssors: predecessor node for each node
    """
    Graph_temp = Graph.copy()
    Graph_temp[[0, start_index]] = Graph_temp[[start_index, 0]]
    Graph_temp[:, [0, start_index]] = Graph_temp[:, [start_index, 0]]
    dist_matrix, predecessors = dijkstra(csgraph=Graph_temp, directed=False, indices=0, return_predecessors=True)

    predecessors[0] = predecessors[start_index]
    predecessors[start_index] = -9999

    temp_0 = predecessors == 0
    temp_start_idx = predecessors == start_index

    predecessors[temp_0] = start_index
    predecessors[temp_start_idx] = 0
    cost = np.sum(dist_matrix)
    return cost, predecessors


def generate_first_guess(list_of_images, gps_coordinates, max_keypoints=25000, use_inliers_only=True,
                         ransac_threshold=10, sift_threshold=0.6, verbose=True, goodmatches_threshold=4,
                         purge_multiples=True, multi_core=False, const_weight_edge=0.1, forced_center_image=False):
    """
    performs the necessary actions to get a first guess stitching solution on a given set of images 
    :param list_of_images: list of images to be stitched
    :param gps_coordinates: gps coordinates of the images
    :param max_keypoints: max nr. of keypoints extracted per image
    :param use_inliers_only: if true, only the inliers of RANSAC provided by findhomography will be used for the bundle adjustment
    :param ransac_threshold: ransac threshold width
    :param sift_threshold: ratio between best/second best match to accept a good match
    :param verbose:
    :param goodmatches_threshold: how many good matches have to be found to consider a connection between 2 images viable
    :param purge_multiples: should world points that have more than 1 point from the same image be purged
    :param multi_core: should the matching be done using multicore.map. Inefficient when using kdetree matcher.
    :param const_weight_edge: constant to be used in the weight calculation for djikstra
    :param forced_center_image: if set to a number, this image will be used as
    :return:
    features: list of the found features of all images
    point_world: list of the assumed world points
    homographies_abs: list of homographies that transform the the image to the center image frame
    center_image: image that will be used as world frame
    edges: list of all connections between pictures with [(i,j,number of matches)]
    predecessors: predecessor for every image, calculated by using djikstra to get the best path to the center image

    """
    # setting up multiprocessing
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    # setting up feature extractor and matcher
    sift = cv2.xfeatures2d.SIFT_create(max_keypoints)
    if verbose:
        print('start feature extraction')
    features = []

    for i in tqdm(range(len(list_of_images)), desc="feature extraction", disable=not verbose):
        img = cv2.cvtColor(list_of_images[i], cv2.COLOR_RGB2GRAY)

        features.append(sift.detectAndCompute(img, None))

    homographies_rel = {}
    point_world_idx = {}
    point_world = []
    edges = []
    Graph = lil_matrix((len(list_of_images), len(list_of_images)))
    tried_connections = set()
    for i in tqdm(range(len(list_of_images)), desc="matching", disable=not verbose):
        results = []
        # calculate distance to the other images
        rel_distance = []
        nr_connections = 0
        for j in range(len(list_of_images)):
            rel_distance.append(geopy.distance.distance(gps_coordinates[i], gps_coordinates[j]).meters)
        closest_images_indices = np.argsort(rel_distance)
        for j in range(len(closest_images_indices) - 1, -1, -1):
            if closest_images_indices[j] == i:
                closest_images_indices = np.delete(closest_images_indices, j)
        closest_images_indices_unvisited = []
        for idx in closest_images_indices:
            if Graph[idx, i] == 0:
                if not (idx, i) in tried_connections:
                    closest_images_indices_unvisited.append(idx)

            else:
                nr_connections += 1
        if verbose:
            print(len(closest_images_indices_unvisited))

        for j in range(math.ceil((len(closest_images_indices_unvisited)) / nprocs)):
            slice = closest_images_indices_unvisited[
                    j * nprocs:min(j * nprocs + nprocs, len(closest_images_indices_unvisited))]
            args = []
            for idx in slice:
                args.append(
                    (i, idx, sift_threshold, use_inliers_only, features, goodmatches_threshold, ransac_threshold))
                tried_connections.add((i, idx))

            # pool.map is only efficient when using brute_force_matcher, if Kdetree is used it ads to much overhead to be worthwile (tripples time used)
            if multi_core:
                results = results + pool.map(mc_matcher, args)
            else:
                results = results + list(map(mc_matcher, args))

            #decides how many imatches should be matched with the current image, depending on how many connections with the current image have allready been found
            nr_good_results = len(results) - results.count(False)
            if len(results) >= 12 and nr_connections + nr_good_results > 20:
                break
            if len(results) >= 20 and nr_connections + nr_good_results > 10:
                break
            if len(results) >= 30 and nr_connections + nr_good_results > 5:
                break
            if len(results) > 40 and nr_connections + nr_good_results > 0:
                break

        for result in results:
            if result != False:
                position = result[0]
                target = result[1]
                matches = result[2]

                Graph[i, target] = const_weight_edge + 1 / len(matches)
                edges.append((i, target, len(matches)))
                homography_rel = result[3]

                try:
                    homographies_rel[(target, position)] = homography_rel
                    homographies_rel[(position, target)] = np.linalg.inv(homography_rel)
                except:
                    print('matrix not invertable:')
                    print(homography_rel)
                    continue

                for match in matches:
                    query = (position, match.queryIdx)
                    train = (target, match.trainIdx)
                    if query not in point_world_idx:
                        if train not in point_world_idx:
                            # creates new point_world if neither keypoint is found in another point_world
                            point_world.append(
                                [(0, 0), set([query, train]), set([position, target]), True])
                            point_world_idx[query] = len(point_world) - 1
                            point_world_idx[train] = len(point_world) - 1
                        else:
                            # if train is found in a point_world, query is added to that point
                            point_world[point_world_idx[train]][1].add(query)
                            if position in point_world[point_world_idx[train]][2]:
                                if verbose:
                                    print('Warning 2 points of same image added to same world point')
                                    print('query')
                                    print(point_world_idx[train])
                                if purge_multiples:
                                    point_world[point_world_idx[train]][3] = False
                            point_world[point_world_idx[train]][2].add(position)
                            point_world_idx[query] = point_world_idx[train]
                    else:
                        if train not in point_world_idx:
                            # if query is found in a point_world,train is added to that point
                            point_world[point_world_idx[query]][1].add(train)
                            if target in point_world[point_world_idx[query]][2]:
                                if verbose:
                                    print('Warning 2 points of same image added to same world point')
                                    print('train')
                                    print(point_world_idx[query])
                                if purge_multiples:
                                    point_world[point_world_idx[query]][3] = False
                            point_world[point_world_idx[query]][2].add(target)
                            point_world_idx[train] = point_world_idx[query]
                        else:
                            # if both query and train are found in 2 different point_worlds those 2 are combined
                            if point_world_idx[query] != point_world_idx[train]:
                                point_world[point_world_idx[query]][1] = point_world[point_world_idx[query]][
                                    1].union(point_world[point_world_idx[train]][1])

                                # check if there are multiple images from for example image 1 in the 2 merged groups
                                sum_of_images = len(point_world[point_world_idx[query]][2]) + \
                                                len(point_world[point_world_idx[train]][2])

                                if sum_of_images > len(point_world[point_world_idx[query]][2]):
                                    if verbose:
                                        print('Warning 2 points of same image added to same world point')
                                        print('merge')
                                        print(point_world_idx[query])
                                    if purge_multiples:
                                        point_world[point_world_idx[query]][3] = False

                                point_world[point_world_idx[query]][2] = point_world[point_world_idx[query]][
                                    2].union(point_world[point_world_idx[train]][2])
                                for point in point_world[point_world_idx[train]][1]:
                                    point_world_idx[point] = point_world_idx[query]
                                point_world[point_world_idx[train]][3] = False

    # discard points that have been marked as bad
    point_world_usable = []
    for point in point_world:
        if point[3]:
            point_world_usable.append(point)
    point_world = point_world_usable

    # chose image with the best connections as center image
    list_of_costs = []
    list_of_predecessors = []
    for i in range(len(list_of_images)):
        cost, predecessors = perform_djikstra(Graph, i)
        list_of_costs.append(cost)
        list_of_predecessors.append(predecessors)
    center_image = np.argmin(list_of_costs)
    if forced_center_image!=False:
        center_image=forced_center_image
    predecessors = list_of_predecessors[center_image]
    if verbose:
        print('center_image is %d' % (center_image))

    # create the absolute homography estimates for all images
    homographies_abs = [None] * len(list_of_images)
    for i in range(len(list_of_images)):
        homographies_abs[i] = homography_path_iter(i, center_image, predecessors, homographies_rel)

    # calculate an estimate for each point_world
    for i in range(len(point_world)):
        est_world_pos = np.zeros((2,))
        for j in range(len(list(point_world[i][1]))):
            point_idx = list(point_world[i][1])[j]
            est_homography = homographies_abs[point_idx[0]]
            start_coord = features[point_idx[0]][0][point_idx[1]].pt
            est_world_pos += apply_homography_to_point(start_coord, est_homography)
        point_world[i][0] = est_world_pos / len(list(point_world[i][1]))

    return features, point_world, homographies_abs, center_image, edges, predecessors


def prepare_data(features, point_world, homographies_abs, camera_matrix, dist_coeffs, verbose=False):
    """
    prepares the data for bundle adjustment
    :param features:the found features of the images
    :param point_world: world points
    :param homographies_abs: absolute homographies for all images
    :param camera_matrix:  camera parameters
    :param dist_coeffs: distortion coefficients
    :param verbose: if true will print out information
    :return:
            x0: first input vector for the bundle adjustment, containing the camera parameters, homographies and world point coordinates
            camera_indices: camera index for every observation
            point_indices: world point index for every observation
            points_camera_frame: coordinates in the original image for every observation
            n_cameras: total number of cameras-> number of images
            n_points: total number of world points
            n_observations: total number of observation. 1 Observation meaning 1 World point visible in 1 Image
    """
    n_cameras = len(homographies_abs)
    n_points = len(point_world)
    observations = np.empty(n_points * 2)
    camera_indices = []
    point_indices = []
    points_camera_frame = []
    camera_params = np.append(camera_matrix.ravel(), dist_coeffs.ravel())

    for i in range(len(point_world)):
        for j in range(len(point_world[i][1])):
            point_idx = list(point_world[i][1])
            camera_indices.append(point_idx[j][0])
            point_indices.append(i)
            points_camera_frame.append(features[point_idx[j][0]][0][point_idx[j][1]].pt)

        observations[2 * i:2 * i + 2] = point_world[i][0].astype(float)

    n_observations = len(points_camera_frame)
    homographies = np.empty(n_cameras * 9)
    for i in range(n_cameras):
        homographies[9 * i:9 * i + 9] = homographies_abs[i].ravel()


    x0 = np.hstack((camera_params.ravel(), homographies.ravel(), observations.ravel()))
    f0 = residuals(x0, n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
    if verbose:
        n = 9 * n_cameras + 2 * n_points + 13
        m = 2 * n_observations
        print("n_cameras: {}".format(n_cameras))
        print("n_World_points: {}".format(n_points))
        print("n_observations: {}".format(n_observations))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        plt.plot(f0)
        plt.title('Residuals before bundle adjustement')
        plt.show()
    camera_indices_array = np.empty((len(camera_indices) * 2))
    camera_indices_array[::2] = camera_indices
    camera_indices_array[1::2] = camera_indices

    return x0, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations


def residuals(params, n_cameras, n_points, n_observations, camera_indicies, point_indicies, points_camera_frame):
    """
    calculates the residual error after undistorting and applying the homographies
    :param params: array containing the camera parameters, homographies and world point coordinates
    :param n_cameras: total number of cameras-> number of images
    :param n_points: total number of world points
    :param n_observations: total number of observation. 1 Observation meaning 1 World point visible in 1 Image
    :param camera_indicies: camera index for every observation
    :param point_indicies: world point index for every observation
    :param points_camera_frame: coordinates in the original image for every observation
    :return: residual error for every observation (x and y axis seperatly) -> length is 2*n_observations
    """
    camera_matrix = params[:9].reshape(3, 3)
    undistort_coeffs = params[9:14]
    homographies = params[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)
    points_world_frame = params[14 + n_cameras * 9:].reshape(n_points, 2)

    point_distorted = np.array(points_camera_frame).astype('float32')
    point_undistorted = cv2.undistortPoints(point_distorted.reshape(-1, int(point_distorted.size / 2), 2), camera_matrix,
                                            undistort_coeffs,
                                            R=np.eye(3), P=camera_matrix)
    residuals = np.empty(n_observations * 2)
    for i in range(n_observations):
        point_warped = apply_homography_to_point(point_undistorted[0][i], homographies[camera_indicies[i]])
        residuals[2 * i:2 * i + 2] = (point_warped - points_world_frame[point_indicies[i]]).reshape(2)
    return residuals


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, center_image,save_matrix=False):
    """
    constructs the sparsity matrix of the jacobian. Can be used to reduce the computational cost of bundle adjustment massively
    adapted from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    :param n_cameras: total number of cameras-> number of images
    :param n_points: total number of world points
    :param camera_indices: camera index for every observation
    :param point_indices: world point index for every observation
    :param center_image: chosen center image
    :param save_matrix: should an image of the sparsity matrix be saved? (will fail for high number of observations, because not enough ram)
    :return: sparsity matrix
    """
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    m = camera_indices.size * 2
    n = 14 + n_cameras * 9 + n_points * 2
    A = lil_matrix((m, n), dtype=int)
    # camera camera_matrixs are same for all observations
    A[:, :14] = 1
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, 14 + camera_indices * 9 + s] = 1
        A[2 * i + 1, 14 + camera_indices * 9 + s] = 1

    for s in range(9):
        A[2 * i, 14 + center_image * 9 + s] = 0
        A[2 * i + 1, 14 + center_image * 9 + s] = 0

    for s in range(2):
        A[2 * i, 14 + n_cameras * 9 + point_indices * 2 + s] = 1
        A[2 * i + 1, 14 + n_cameras * 9 + point_indices * 2 + s] = 1
    plt.spy(A)
    plt.show()
    if save_matrix:
        try:
            cv2.imwrite('sparse.png', A.toarray() * 255)
        except:
            print('problem while writing sparse matrix')
    return A


def closest_image_map(height_ori, width_ori, transformed_centers, trans_img_corners, downscaling_factor=4,
                      verbose=False):
    '''
    generates image in which the closest origin image to pixels is indicated
    :param height_ori: height of the stitched output
    :param width_ori: width of the stitched output
    :param transformed_centers: (transformed) centerpoints of the images
    :param trans_img_corners: (transformed) cornerpoints of the images
    :param downscaling_factor:  downscaling factor for the calculation
    :param verbose: if true some information will be given in the console
    :return: Image in which the pixel value indicates the closest image. If a pixel is 1 it indicates that the first image in coordinates is the closest
    '''
    if verbose:
        print('start calculating closest image map')
        start = time.time()
    height = int(round(height_ori / downscaling_factor))
    width = int(round(width_ori / downscaling_factor))
    new_trans_img_corners = []

    coordinates = []
    for coordinate in transformed_centers:
        coordinates.append((round(coordinate[1] / downscaling_factor), round(coordinate[0] / downscaling_factor)))

    for image in trans_img_corners:
        corner_array = np.zeros((4, 2))
        for i in range(len(image)):
            corner_array[i, :] = (round(image[i][1] / downscaling_factor), round(image[i][0] / downscaling_factor))
        new_trans_img_corners.append(corner_array)
    trans_img_corners = new_trans_img_corners

    a = np.zeros((height, width))
    x = np.arange(0, a.shape[0], 1)
    y = np.arange(0, a.shape[1], 1)
    xx, yy = np.meshgrid(y, x)
    shortest_distance = np.full_like(a, math.sqrt(height ** 2 + width ** 2), dtype=np.single)
    node = np.zeros_like(shortest_distance, dtype=np.uint8)

    for i in range(0, len(trans_img_corners)):
        y_lower = int(max(0, round(np.min(trans_img_corners[i][:, 0]))))
        y_upper = int(round(np.max(trans_img_corners[i][:, 0] + 1)))
        x_lower = int(max(0, round(np.min(trans_img_corners[i][:, 1]))))
        x_upper = int(round(np.max(trans_img_corners[i][:, 1] + 1)))
        shortest_distance_roi = shortest_distance[y_lower:y_upper, x_lower:x_upper]
        node_roi = node[y_lower:y_upper, x_lower:x_upper]
        xx_roi = xx[y_lower:y_upper, x_lower:x_upper]
        yy_roi = yy[y_lower:y_upper, x_lower:x_upper]

        trans_img_corners_roi = np.empty_like(trans_img_corners[i])
        trans_img_corners_roi[:, 0] = trans_img_corners[i][:, 1] - x_lower
        trans_img_corners_roi[:, 1] = trans_img_corners[i][:, 0] - y_lower
        image_matrix = np.zeros_like(shortest_distance_roi)
        cv2.fillPoly(image_matrix, [trans_img_corners_roi.reshape(-1, 1, 2).astype('int')], 1)

        shortest_distance_2_roi = np.sqrt((yy_roi - coordinates[i][0]) ** 2 + (xx_roi - coordinates[i][1]) ** 2)

        shorter_matrix = (shortest_distance_roi > shortest_distance_2_roi) * (image_matrix == 1)
        shortest_distance_roi[shorter_matrix] = shortest_distance_2_roi[shorter_matrix]
        node_roi[shorter_matrix] = i + 1

    upscale = cv2.resize(node, (width_ori, height_ori), interpolation=cv2.INTER_AREA)
    if verbose:
        print('generating the closest image map took %d seconds' % (time.time() - start))
        plt.imshow(upscale)
        plt.colorbar()
        plt.show()
    return upscale


def apply_homography_to_point(point, M):
    '''
    applies homography to a point
    :param point: point given in coordinates
    :param M: homography matrix
    :return: transformed point in coordinates
    '''
    a = np.array([np.asarray(point)], dtype='float32').reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(a, M)
    return result.reshape(1, 2)[0]


def gauss_kernel(sigma):
    '''
    defines gauss filter for a given sigma
    :param sigma: sigma that is used to generate the kernel
    :return: kernel of size 3 sigma x 3 sigma
    '''
    # calculating the needed size of the kernel for the given sigma
    size = int(2 * np.ceil(4 * sigma) + 1)
    # creating meshgrid for x and y
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    # calculating the kernel
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    # normalize the kernel to 1
    kernel = (kernel / kernel.sum()).astype('float64')
    return kernel


def combine_images(list_of_images, homographies, verbose=False, edges=False, predecessors=False,center_image=0,
                   perform_blending=True, sigma=5, k=3):
    """
    combines images based on the provided homographies
    :param list_of_images: the images that should be combined
    :param homographies: the absolute homographies for all images
    :param verbose: if True will print out information
    :param edges: the found edges between images, if both edges and predecessors are provided, a visualization of the graph will be created.
    :param predecessors: predecessor for every image, calculated by using djikstra to get the best path to the center image
    :param center_image: center_image, only affects hich image is marked as center in the visualization
    :param perform_blending: if True will use multi band blending at the image borders. This might take a long time depending on the size of the combined image and the number of images
    :param sigma: base sigma value for the multi band blending
    :param k: number of bands to be used in the blending
    :return:
    end_result: combined image
    downscaling_factor: for memory reasons the size of the combined image is limited to 10^8 pixels, are the combined images larger than that, the result will be downscaled by this downscaling factor.
    trans_img_corner_center_image: transformed upper left corner of the center image (origin of the world frame)

    """
    w = list_of_images[0].shape[1]
    h = list_of_images[0].shape[0]

    # calculate the transformed middle coordinates
    trans_img_centers = []
    trans_img_corners = []
    for homography in homographies:

        # center = apply_homography_to_point((offset_x * 0.5, offset_y * 0.5), np.matmul(offset, homography))
        center = apply_homography_to_point((w * 0.5, h * 0.5), homography)
        trans_img_centers.append(list(center))

        # delete image edges to avoid artifacts created by the undistortion (black at the edges)
        undistort_offset = 100
        corner1 = apply_homography_to_point((undistort_offset, undistort_offset), homography)
        corner2 = apply_homography_to_point((w - undistort_offset, undistort_offset), homography)
        corner3 = apply_homography_to_point((w - undistort_offset, h - undistort_offset), homography)
        corner4 = apply_homography_to_point((undistort_offset, h - undistort_offset), homography)
        trans_img_corners.append((list(corner1), list(corner2), list(corner3), list(corner4)))

        # calculate the estimated max and min of the endresult:
        y_max = trans_img_corners[0][0][1]
        y_min = trans_img_corners[0][0][1]
        x_max = trans_img_corners[0][0][0]
        x_min = trans_img_corners[0][0][0]
        for img in trans_img_corners:
            for corner in img:
                y_max = int(round(max(y_max, corner[1])))
                y_min = int(round(min(y_min, corner[1])))
                x_max = int(round(max(x_max, corner[0])))
                x_min = int(round(min(x_min, corner[0])))

        w_res = x_max - x_min
        h_res = y_max - y_min

    offset_y = int(round(abs(y_min)))
    offset_x = int(round(abs(x_min)))
    for i in range(len(trans_img_centers)):

        trans_img_centers[i][1] += offset_y
        trans_img_centers[i][0] += offset_x
        for corner in trans_img_corners[i]:
            corner[1] += offset_y
            corner[0] += offset_x



    offset = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

    closest_img_map = closest_image_map(h_res, w_res, trans_img_centers, trans_img_corners, downscaling_factor=4,
                                        verbose=True)
    # if the resulting image would not fit into memory, we produce a downscaled version
    downscaling_factor = 1
    if (h_res * w_res / downscaling_factor ** 2 > 10000 ** 2):
        downscaling_factor = math.sqrt(h_res * w_res / 10000 ** 2)

    trans_img_center_center_image = np.array(trans_img_centers[center_image])/downscaling_factor
    trans_img_corner_center_image=np.array([trans_img_corners[center_image][0][0],trans_img_corners[center_image][0][1]])/downscaling_factor

    h_res_ds = int(round(h_res / downscaling_factor))
    w_res_ds = int(round(w_res / downscaling_factor))
    closest_img_map = cv2.resize(closest_img_map, (w_res_ds, h_res_ds))
    # perform image blending
    if perform_blending:

        result = [np.zeros((h_res_ds, w_res_ds, 3), dtype='float32')] * (k + 1)
        normalization = [np.zeros((h_res_ds, w_res_ds), dtype='float32')] * (k + 1)

        for i in tqdm(range(0, len(homographies)), desc="blending images", disable=not verbose):

            # add the connected images
            M = np.matmul(offset, homographies[i])
            warped_image = cv2.warpPerspective(list_of_images[i], M,
                                               (w_res, h_res))
            warped_image = cv2.resize(warped_image, (w_res_ds, h_res_ds))

            # multi_band_blending according to http://matthewalunbrown.com/papers/ijcv2007.pdf
            x, y, dx, dy = cv2.boundingRect((closest_img_map == i + 1).astype('uint8'))
            roi_offset = int(math.floor(3 * sigma * math.sqrt(2 * k + 1)))
            roi = (
                max(0, y - roi_offset), min(h_res_ds, y + dy + roi_offset), max(0, x - roi_offset),
                min(w_res_ds, x + dx + roi_offset))
            mask = np.zeros((roi[1] - roi[0], roi[3] - roi[2]), dtype='float32')
            mask[closest_img_map[roi[0]:roi[1], roi[2]: roi[3]] == i + 1] = 1

            kernel = gauss_kernel(sigma)
            I = [cv2.filter2D(warped_image[roi[0]:roi[1], roi[2]: roi[3]], ddepth=cv2.CV_32F, kernel=kernel)]
            B = [warped_image[roi[0]:roi[1], roi[2]: roi[3]] - I[0]]
            W = [cv2.filter2D(mask, ddepth=cv2.CV_32F, kernel=kernel)]
            normalization[0][roi[0]:roi[1], roi[2]: roi[3]] += W[-1]
            for l in range(3):
                result[0][roi[0]:roi[1], roi[2]: roi[3], l] += B[0][:, :, l] * W[0]
            for j in range(1, k + 1):
                kernel = gauss_kernel(sigma * math.sqrt(2 * (j) + 1))
                I.append(cv2.filter2D(I[-1], ddepth=cv2.CV_32F, kernel=kernel))
                if j == k:
                    B.append(I[-2])
                else:
                    B.append(I[-2] - I[-1])
                W.append(cv2.filter2D(W[-1], ddepth=cv2.CV_32F, kernel=kernel))
                normalization[j][roi[0]:roi[1], roi[2]: roi[3]] += W[-1]
                for l in range(3):
                    result[j][roi[0]:roi[1], roi[2]: roi[3], l] += B[j][:, :, l] * W[j]
        end_result = np.zeros_like(result[0])
        for i in range(k + 1):
            normalization[i][normalization == 0] = np.inf
            for l in range(3):
                end_result[:, :, l] += result[i][:, :, l] / normalization[i]
        end_result[closest_img_map == 0] = 0
    else:
        end_result = np.zeros((h_res_ds, w_res_ds, 3), dtype='uint8')
        for i in tqdm(range(0, len(homographies)), desc="combining images", disable=not verbose):
            M = np.matmul(offset, homographies[i])
            warped_image = cv2.warpPerspective(list_of_images[i], M,
                                               (w_res, h_res))
            warped_image = cv2.resize(warped_image, (w_res_ds, h_res_ds))
            end_result[closest_img_map == i + 1] = warped_image[closest_img_map == i + 1]

    # create visualization
    if not edges == False and type(predecessors) == np.ndarray:
        graph_visualization = end_result.copy()
        for i in range(len(trans_img_centers)):
            trans_img_centers[i] = tuple(int(round(n / downscaling_factor)) for n in trans_img_centers[i])
            # show outlines of picture
            thresh = closest_img_map == i + 1
            im2, contours, hierarchy = cv2.findContours(thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(graph_visualization, contours, -1, (255, 255, 255), 10)

        line_thickness_per_match = 190 / np.max(np.array(edges)[:, 2])

        # display other edges in blue
        for edge in edges:
            position = edge[0]
            target = edge[1]
            nr_matches = edge[2]
            line_thickness = 10 + int(math.ceil(line_thickness_per_match * nr_matches))
            if not (predecessors[position] == target or predecessors[target] == position):
                color = (255, 0, 0)
                cv2.line(graph_visualization, trans_img_centers[position], trans_img_centers[target], color,
                         line_thickness)

        # display shortest paths in green
        for edge in edges:
            position = edge[0]
            target = edge[1]
            nr_matches = edge[2]
            line_thickness = 10 + int(math.ceil(line_thickness_per_match * nr_matches))
            if predecessors[position] == target or predecessors[target] == position:
                color = (0, 255, 0)
                cv2.line(graph_visualization, trans_img_centers[position], trans_img_centers[target], color,
                         line_thickness)
        # display nodes
        for i in range(len(trans_img_centers)):
            if i == center_image:
                radius = 250
            else:
                radius = 100
            cv2.circle(graph_visualization, trans_img_centers[i], radius, (0, 0, 255), -1)

            CENTER = trans_img_centers[i]
            text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_COMPLEX, 5, 10)[0]

            text_origin = (int(CENTER[0] - text_size[0] / 2), int(CENTER[1] + text_size[1] / 2))

            cv2.putText(graph_visualization, str(i), text_origin, cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0), 10)

        return end_result, downscaling_factor,trans_img_corner_center_image,trans_img_center_center_image, graph_visualization

    return end_result, downscaling_factor,trans_img_corner_center_image,trans_img_center_center_image


def homography_from_rotation(yaw, pitch, roll, camera_matrix):
    """
    constructs a homography matrix from yaw, pitch  and roll rotations.
    the rotations are applied in the following order: roll->pitch->yaw
    :param yaw: yaw angle in rad
    :param pitch: pitch angle in rad
    :param roll: roll angle in rad
    :param camera_matrix: camera camera_matrix matrix
    :return: rotation homography matrix
    """
    R_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R_roll = np.array([[math.cos(roll), 0, math.sin(roll)], [0, 1, 0], [-math.sin(roll), 0, math.cos(roll)]])
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(R_yaw, R_pitch), R_roll)
    RK = np.matmul(R, np.linalg.inv(camera_matrix))
    M = np.matmul(camera_matrix, RK)
    return M


def correct_pose_Homography(yaw_gimbal, pitch_gimbal, roll_gimbal, camera_matrix):
    """
    takes the angles read out from image metadata and creates a correction homography matrix
    :param yaw_gimbal: yaw angle in degree, 0 degree meaning north
    :param pitch_gimbal: pitch angle in degree, -90 degree meaning looking top down
    :param roll_gimbal: roll angle in degree, 0 degree meaning a horizontal orientation
    :param camera_matrix: camera camera_matrix matrix
    :return: correction homography matrix
    """
    yaw_gimbal = math.radians(yaw_gimbal)
    pitch_gimbal = math.radians((90 + pitch_gimbal))
    roll_gimbal = math.radians(roll_gimbal)
    M = homography_from_rotation(yaw_gimbal, pitch_gimbal, roll_gimbal, camera_matrix)
    return M


def estimate_scale(xmp_metadata,  w, fov, center_image):
    """
    estimates a scaling factor between real world and the center image, based on the image metadata
    :param xmp_metadata: metadata of all images
    :param w: width of an image
    :param fov: horizontal field of view of the camera
    :param center_image: index of the center image
    :return: scaling factor in m/px
    """
    fov = math.radians(fov)
    altitude = float(xmp_metadata[center_image]['Xmp.drone-dji.RelativeAltitude'])
    roll_gimbal = float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalRollDegree'])
    pitch_gimbal = float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalPitchDegree'])
    pitch_gimbal = math.radians(abs(-90 - pitch_gimbal))
    roll_gimbal = math.radians(abs(roll_gimbal))
    # correct for roll and pitch under the assumption, that the field is a perfectly straight plane
    distance_pitch_compensated = altitude / math.cos(pitch_gimbal)
    distance_roll_compensated = distance_pitch_compensated / math.cos(roll_gimbal)
    horizontal_distance = distance_roll_compensated * math.tan(fov / 2) * 2
    scale = horizontal_distance / w  # m/px of the original image
    return scale


def perform_stitching(img_dir, max_keypoints, perform_blending=True, ransac_threshold=10, sift_threshold=0.6,forced_center_image=False,show_residuals=False):
    """
    stitches the images in the img_dir together nad saves them in working directory/results/YYYYMMDD HHMM (date and time)
    :param img_dir: directory of the iamges to be stitched
    :param max_keypoints: number of keypoints per image
    :param perform_blending: if True the images will be blended using multi band blending (time intensive)
    :param ransac_threshold: Ransac threshold width
    :param sift_threshold: ratio between best/second best match to accept a good match
    :param forced_center_image: if set, the center image will be forced to this number
    :return:
    """
    start=time.time()
    now = datetime.now()
    results_dir = now.strftime(img_dir+"/%Y%m%d %H%M")
    os.mkdir(results_dir)

    data_path = os.path.join(img_dir, '*jpg')
    files = glob.glob(data_path)
    f = open(results_dir + '/log.txt', "w+")

    # oneplus 7 pro estimated parameters
    # camera_matrix = np.array([[4347.358087366480, 0, 1780.759210199340], [0, 4349.787712956160, 1518.540335312340], [0, 0, 1]])
    # distCoeffs = np.array([0.0928, -0.7394, 0, 0])

    # spark parameters estimation read out from pix4d
    camera_matrix = np.array([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
    distCoeffs = np.array([0.117, -0.298, 0.001, 0, 0.1420])

    data = []
    gps_coordinates = []
    xmp_metadata = []
    data_undistorted = []
    patch_cv2_pickiling()
    for f1 in files:
        img1 = cv2.imread(f1)
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        data.append(img1)
        gps_coord = gpsphoto.getGPSData(f1)
        metadata = pyexiv2.Image(f1)
        xmp_metadata.append(metadata.read_xmp())
        gps_coordinates.append((gps_coord['Latitude'], gps_coord['Longitude']))

    w = data[0].shape[1]

    # choosing ransac threshold very high to get inliers that don't show up because of distortion (points near the edges)
    features, point_world, homographies_abs, center_image, edges, predecessors = generate_first_guess(data,
                                                                                                      gps_coordinates,
                                                                                                      max_keypoints=max_keypoints,
                                                                                                      sift_threshold=sift_threshold,
                                                                                                      ransac_threshold=ransac_threshold,
                                                                                                      use_inliers_only=True,
                                                                                                      goodmatches_threshold=10,
                                                                                                      purge_multiples=False, forced_center_image=forced_center_image)
    print('generated matrixes')

    for i in range(len(data)):
        data_undistorted.append(cv2.undistort(data[i], camera_matrix, distCoeffs))

    x0, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations = prepare_data(
        features, point_world, homographies_abs, camera_matrix, distCoeffs, verbose=True)

    res_image, downscaling_factor,trans_img_corner,trans_img_center, graph_visu = combine_images(data, homographies_abs, verbose=True, edges=edges, predecessors=predecessors,
                                              perform_blending=False, center_image=center_image)
    cv2.imwrite(results_dir + '/res_image_guess.jpg', res_image)
    cv2.imwrite(results_dir + '/res_image_guess_visu.jpg', graph_visu)

    if show_residuals:

        resid=residuals(x0,n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
        points=x0[14 + n_cameras * 9:].reshape(n_points,1,2)
        resid_array=np.zeros((n_points))
        for i in range(int(len(resid)/2)):
            resid_array[point_indices[i]]+=abs(resid[2*i])
            resid_array[point_indices[i]] += abs(resid[2*i+1])
        for i in range(len(points)):
            cv2.circle(res_image,tuple((points[i][0]/downscaling_factor+trans_img_corner).astype("int")),10,(0,max(0,int(255-2*resid_array[i])),min(255,int(2*resid_array[i]))),-1)
        cv2.imwrite(results_dir + '/res_image_guess_residuals.jpg', res_image)




    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, center_image)
    res = least_squares(residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, xtol=1e-8, method='trf',
                        max_nfev=200,
                        args=(n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame))
    print('performed bundle adjustement')
    plt.plot(res.fun)
    plt.title('Residuals after bundle adjustement')
    plt.show()
    print('mean residuals= %f' % (np.mean(np.abs(res.fun))))
    camera_matrix = res.x[:9].reshape(3, 3)
    distCoeffs = res.x[9:14]
    homographies = res.x[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)

    for i in range(len(data)):
        data_undistorted[i] = cv2.undistort(data[i], camera_matrix, distCoeffs)

    res_image, downscaling_factor,trans_img_corner,trans_img_center, graph_visu = combine_images(data_undistorted, homographies, verbose=True, edges=edges,
                                              predecessors=predecessors, perform_blending=False, center_image=center_image)
    cv2.imwrite(results_dir + '/res_image.jpg', res_image)
    cv2.imwrite(results_dir + '/res_image_visu.jpg', graph_visu)

    if show_residuals:

        resid=residuals(res.x,n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
        points=res.x[14 + n_cameras * 9:].reshape(n_points,1,2)
        resid_array=np.zeros((n_points))
        for i in range(int(len(resid)/2)):
            resid_array[point_indices[i]]+=abs(resid[2*i])
            resid_array[point_indices[i]] += abs(resid[2*i+1])
        for i in range(len(points)):
            cv2.circle(res_image,tuple((points[i][0]/downscaling_factor+trans_img_corner).astype("int")),10,(0,max(0,int(255-2*resid_array[i])),min(255,int(2*resid_array[i]))),-1)
        cv2.imwrite(results_dir + '/res_image_residuals.jpg', res_image)

    print('camera_matrix matrix is:')
    print(camera_matrix)
    print('distortion coeffs are:')
    print(distCoeffs)

    # correct for angles
    roll_gimbal = float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalRollDegree'])
    pitch_gimbal = float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalPitchDegree'])
    yaw_gimbal = float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalYawDegree'])

    pose_correction_matrix = correct_pose_Homography(yaw_gimbal, pitch_gimbal, roll_gimbal, camera_matrix)
    homograpies_abs_pose_corr = []
    for M in homographies:
        homograpies_abs_pose_corr.append(np.matmul(pose_correction_matrix, M))

    res_image, downscaling_factor,trans_img_corner,trans_img_center, graph_visu = combine_images(data_undistorted, homograpies_abs_pose_corr, verbose=True,
                                                               edges=edges, predecessors=predecessors,
                                                               perform_blending=perform_blending,
                                                               center_image=center_image)
    cv2.imwrite(results_dir + '/res_image_oriented.jpg', res_image)
    cv2.imwrite(results_dir + '/res_image_oriented_visu.jpg', graph_visu)

    scale = estimate_scale(xmp_metadata, w, 66.55, center_image)
    print(scale)
    print(scale * downscaling_factor)
    end=time.time()
    runtime=end-start

    f.write("ransac_threshold is: %f \r\n" % (ransac_threshold))
    f.write("sift_threshold is: %f \r\n" % (sift_threshold))
    f.write("blending enabled: %s \r\n" % (perform_blending))
    f.write('matches per image: %d \r\n' % (max_keypoints))
    f.write('center image is: %d \r\n'%(center_image))
    f.write('center image center coordinates:%d,%d\r\n'%(trans_img_center[0],trans_img_center[1]))
    f.write('gps location at this coordinates: lat = %f, lon = %f \r\n'%(gps_coordinates[center_image][0],gps_coordinates[center_image][1]))
    f.write("scale is: %f m/px\r\n" % (scale * downscaling_factor))
    f.write("runtime was : "+ time.strftime('%H:%M:%S', time.gmtime(runtime)))
    f.close()

    return results_dir + '/res_image_oriented.jpg',scale*downscaling_factor


if __name__ == '__main__':
    # %%

    from random import randrange

    max_keypoints = 10000
    #img_dir = r"C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\photos25m"  # Enter Directory of all images
    #perform_stitching(img_dir, max_keypoints, perform_blending=True, ransac_threshold=10, sift_threshold=0.6)
    img_dir = r"C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\photos5"  # Enter Directory of all images
    perform_stitching(img_dir, max_keypoints, perform_blending=False, ransac_threshold=10, sift_threshold=0.6,show_residuals=True)
