import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sys
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

import copyreg


def patch_cv2_pickiling():
    """
    makes cv2 keypoints and dmatches pickable
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
    #setting up the matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # extracting the keypoints
    all_matches = flann.knnMatch(features[position][1], features[target][1], k=2)
    goodmatches = []
    for m, n in all_matches:
        if m.distance < sift_threshold * n.distance:
            goodmatches.append(m)

    print("process %d: matching %d and %d got %d matches" % (os.getpid(), position, target, len(goodmatches)))

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
    if len(matches)<4:
        return False
    return (position, target, matches, M)


def homography_path_iter(position, target, predecessors, homographys_rel, homography=np.eye(3)):
    if position == target:
        return homography
    elif predecessors[position] == -9999:
        print("one or more images are not connected to the central image")
        print("central image index is:%d" % (target))
        print("failed to connect image with index: %d" % (position))
        return False
    else:
        homography = np.matmul(homographys_rel[(predecessors[position], position)], homography)

        return homography_path_iter(predecessors[position], target, predecessors, homographys_rel, homography)

    return result


def perform_djikstra(Graph, start_index):
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


def generate_matrixes(list_of_images, gps_coordinates, MAX_MATCHES=5000, use_inliers_only=True,
                      ransac_threshold=5, sift_threshold=0.7, verbose=True, goodmatches_threshold=4,
                      purge_multiples=True):
    # setting up multiprocessing
    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    # setting up feature extractor and matcher
    sift = cv2.xfeatures2d.SIFT_create(MAX_MATCHES)
    if verbose:
        print('start feature extraction')
    features = []

    for i in tqdm(range(len(list_of_images)), desc="feature extraction", disable=not verbose):
        img = cv2.cvtColor(list_of_images[i], cv2.COLOR_RGB2GRAY)

        features.append(sift.detectAndCompute(img, None))

    matches = {}
    homographys_rel = {}
    point_world_idx = {}
    point_world = []
    Graph = lil_matrix((len(list_of_images), len(list_of_images)))
    tried_connections=set()
    for i in tqdm(range(len(list_of_images)), desc="matching", disable=not verbose):
        results = []
        # calculate distance to the other images
        rel_distance = []
        nr_connections=0
        for j in range(len(list_of_images)):
            rel_distance.append(geopy.distance.distance(gps_coordinates[i], gps_coordinates[j]).meters)
        closest_images_indices = np.argsort(rel_distance)
        for j in range(len(closest_images_indices) - 1, -1, -1):
            if closest_images_indices[j] == i:
                closest_images_indices = np.delete(closest_images_indices, j)
        closest_images_indices_unvisited = []
        for idx in closest_images_indices:
            if Graph[idx, i] == 0:
                if not (idx,i) in tried_connections:
                    closest_images_indices_unvisited.append(idx)

            else:
                nr_connections+=1
        print(len(closest_images_indices_unvisited))

        for j in range(math.ceil((len(closest_images_indices_unvisited)) / nprocs)):
            slice = closest_images_indices_unvisited[
                    j * nprocs:min(j * nprocs + nprocs, len(closest_images_indices_unvisited))]
            args = []
            for idx in slice:
                args.append(
                    (i, idx, sift_threshold, use_inliers_only, features, goodmatches_threshold, ransac_threshold))
                tried_connections.add((i,idx))
            results = results + pool.map(mc_matcher, args)

            nr_good_results=len(results)-results.count(False)
            if nr_connections + nr_good_results > 8:
                break
            if len(results)>=12 and nr_connections+nr_good_results>4:
                break
            if len(results) >= 20 and nr_connections + nr_good_results > 1:
                break
            if len(results)>40 and nr_connections+nr_good_results>0:
                break

        for result in results:
            if result != False:
                position = result[0]
                target = result[1]
                matches = result[2]
                Graph[i, target] = 1 / len(matches)
                homography_rel = result[3]

                try:
                    homographys_rel[(target, position)] = homography_rel
                    homographys_rel[(position, target)] = np.linalg.inv(homography_rel)
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
    predecessors = list_of_predecessors[np.argmin(list_of_costs)]
    center_image = np.argmin(list_of_costs)
    print('center_image is %d' % (center_image))

    # create the absolute homography estimates for all images
    homographys_abs = [None] * len(list_of_images)
    for i in range(len(list_of_images)):
        homographys_abs[i] = homography_path_iter(i, center_image, predecessors, homographys_rel)

    # calculate an estimate for each point_world
    for i in range(len(point_world)):
        est_world_pos = np.zeros((2,))
        for j in range(len(list(point_world[i][1]))):
            point_idx = list(point_world[i][1])[j]
            est_homography = homographys_abs[point_idx[0]]
            start_coord = features[point_idx[0]][0][point_idx[1]].pt
            est_world_pos += apply_homography_to_point(start_coord, est_homography)
        point_world[i][0] = est_world_pos / len(list(point_world[i][1]))

    return features, point_world, homographys_abs,center_image


def prepare_data(features, point_world, homographys_abs, intrinsic, dist_coeffs, outlier_threshold=100, verbose=False):
    n_cameras = len(homographys_abs)
    n_points = len(point_world)
    points_world_frame = np.empty(n_points * 2)
    camera_indices = []
    point_indices = []
    points_camera_frame = []
    camera_params = np.append(intrinsic.ravel(), dist_coeffs.ravel())

    for i in range(len(point_world)):
        for j in range(len(point_world[i][1])):
            point_idx = list(point_world[i][1])
            camera_indices.append(point_idx[j][0])
            point_indices.append(i)
            points_camera_frame.append(features[point_idx[j][0]][0][point_idx[j][1]].pt)

        points_world_frame[2 * i:2 * i + 2] = point_world[i][0].astype(float)

    n_observations = len(points_camera_frame)
    homographys = np.empty(n_cameras * 9)
    for i in range(n_cameras):
        homographys[9 * i:9 * i + 9] = homographys_abs[i].ravel()

    # before outlier removal

    x0 = np.hstack((camera_params.ravel(), homographys.ravel(), points_world_frame.ravel()))
    f0 = residuals(x0, n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
    if verbose:
        n = 9 * n_cameras + 2 * n_points + 13
        m = 2 * n_observations
        print("before outlier removal")
        print("n_cameras: {}".format(n_cameras))
        print("n_World_points: {}".format(n_points))
        print("n_observations: {}".format(n_observations))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        plt.plot(f0)
        plt.title('Residuals before outlier removal')
        plt.show()
    camera_indices_array = np.empty((len(camera_indices) * 2))
    camera_indices_array[::2] = camera_indices
    camera_indices_array[1::2] = camera_indices

    #todo: remove obsolete code
    """
    camera_residual_mean = []
    camera_residual_sigma = []
    for i in range(n_cameras):
        dataset = f0[camera_indices_array == i]
        camera_residual_mean.append(np.median(dataset))
        camera_residual_sigma.append(np.std(dataset))

    for i in range(n_observations - 1, -1, -1):
        camera_idx = camera_indices[i]
        upper=camera_residual_mean[camera_idx] + 3 * camera_residual_sigma[camera_idx]
        lower=camera_residual_mean[camera_idx] - 3 * camera_residual_sigma[camera_idx]

        if upper < f0[2 * i] or lower>f0[2 * i] or upper < f0[2 * i+1] or lower>f0[2 * i+1]:
            camera_indices.pop(i)
            point_indices.pop(i)
            points_camera_frame.pop(i)

    n_observations = len(points_camera_frame)
    # after outlier removal
    if verbose:
        n = 9 * n_cameras + 2 * n_points + 13
        m = 2 * n_observations
        print("after outlier removal")
        print("n_cameras: {}".format(n_cameras))
        print("n_World_points: {}".format(n_points))
        print("n_observations: {}".format(n_observations))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))

        x1 = np.hstack((camera_params.ravel(), homographys.ravel(), points_world_frame.ravel()))
        f1 = residuals(x1, n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
        plt.plot(f1)
        plt.title('Residuals after outlier removal')
        plt.show()"""

    return camera_params, points_world_frame, homographys, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations


def residuals(params, n_cameras, n_points, n_observations, camera_indicies, point_indicies, points_camera_frame):
    intrinsic = params[:9].reshape(3, 3)
    undistort_coeffs = params[9:14]
    homographys = params[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)
    points_world_frame = params[14 + n_cameras * 9:].reshape(n_points, 2)

    point_distorted = np.array(points_camera_frame).astype('float32')
    point_undistorted = cv2.undistortPoints(point_distorted.reshape(-1, int(point_distorted.size / 2), 2), intrinsic,
                                            undistort_coeffs,
                                            R=np.eye(3), P=intrinsic)
    residuals = np.empty(n_observations * 2)
    for i in range(n_observations):
        point_warped = apply_homography_to_point(point_undistorted[0][i], homographys[camera_indicies[i]])
        residuals[2 * i:2 * i + 2] = (point_warped - points_world_frame[point_indicies[i]]).reshape(2)
    return residuals


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,center_image):
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    m = camera_indices.size * 2
    n = 14 + n_cameras * 9 + n_points * 2
    A = lil_matrix((m, n), dtype=int)
    # camera intrinsics are same for all observations
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
    fig = plt.figure()
    plt.spy(A)
    plt.show()
    try:
        cv2.imwrite('sparse.png',A.toarray()*255)
    except:
        print('problem while writing sparse matrix')
    return A


def closest_image_map(height_ori, width_ori, coordinates, trans_img_corners, h_img, w_img, downscaling_factor=4,
                      margin=1,
                      verbose=False):
    '''
    generates image in which the closest origin image to pixels is indicated
    :param height_ori: height of the stitched output
    :param width_ori: width of the stitched output
    :param trans_img_corners: (transformed) cornerpoints of the image
    :param h_img: height of one image
    :param w_img: width of image
    :param downscaling_factor:  downscaling factor for the calculation
    :param margin: margin around pictures that is still calculated, expressed in image_size
    :param verbose: if true some information will be given in the console
    :return: Image in which the pixel value indicates the closest image. If a pixel is 1 it indicates that the first image in coordinates is the closest
    '''
    if verbose:
        print('start calculating closest image map')
        start = time.time()
    height = int(round(height_ori / downscaling_factor))
    width = int(round(width_ori / downscaling_factor))
    new_trans_img_corners = []

    new_coordinates = []
    for coordinate in coordinates:
        new_coordinates.append((round(coordinate[1] / downscaling_factor), round(coordinate[0] / downscaling_factor)))
    coordinates = new_coordinates

    for image in trans_img_corners:
        corner_array = np.zeros((4, 2))
        for i in range(len(image)):
            corner_array[i, :] = (round(image[i][1] / downscaling_factor), round(image[i][0] / downscaling_factor))
        new_trans_img_corners.append(corner_array)
    trans_img_corners = new_trans_img_corners
    h_img = round(h_img / downscaling_factor)
    w_img = round(w_img / downscaling_factor)

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


def multiple_v3(list_of_images, homographys, margin=1):
    w = list_of_images[0].shape[1]
    h = list_of_images[0].shape[0]

    # calculate the transformed middle coordinates
    trans_img_centers = []
    for homography in homographys:
        # center = apply_homography_to_point((offset_x * 0.5, offset_y * 0.5), np.matmul(offset, homography))
        center = apply_homography_to_point((w * 0.5, h * 0.5), homography)
        trans_img_centers.append(list(center))

        # calculate the estimated max and min of the endresult:
        y_max = 0.5 * h + margin * h
        y_min = 0.5 * h - margin * h
        x_max = 0.5 * w + margin * w
        x_min = 0.5 * w - margin * w
        for img in trans_img_centers:
            y_max = int(round(max(y_max, img[1] + margin * h)))
            y_min = int(round(min(y_min, img[1] - margin * h)))
            x_max = int(round(max(x_max, img[0] + margin * w)))
            x_min = int(round(min(x_min, img[0] - margin * w)))

        w_res = x_max - x_min
        h_res = y_max - y_min

    offset_y = int(round(abs(y_min)))
    offset_x = int(round(abs(x_min)))
    for i in range(len(trans_img_centers)):
        trans_img_centers[i][1] += offset_y
        trans_img_centers[i][0] += offset_x

    offset = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

    closest_img_map = closest_image_map(h_res, w_res, trans_img_centers, h, w, downscaling_factor=4, verbose=True)

    result = np.zeros((h_res, w_res, 3), dtype='uint8')
    for i in range(0, len(homographys)):
        # add the connected images
        M = np.matmul(offset, homographys[i])
        warped_image = cv2.warpPerspective(list_of_images[i], M,
                                           (result.shape[1], result.shape[0]))
        warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
        result[result == 0] = warped_image[result == 0]
        result[np.logical_and(closest_img_map == i + 1, warped_gray != 0)] = warped_image[
            np.logical_and(closest_img_map == i + 1, warped_gray != 0)]
    return result


# laplacian blending(not used)

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


def multiple_v4(list_of_images, homographys, margin=1, verbose=False):
    w = list_of_images[0].shape[1]
    h = list_of_images[0].shape[0]

    # calculate the transformed middle coordinates
    trans_img_centers = []
    trans_img_corners = []
    for homography in homographys:

        # center = apply_homography_to_point((offset_x * 0.5, offset_y * 0.5), np.matmul(offset, homography))
        center = apply_homography_to_point((w * 0.5, h * 0.5), homography)
        trans_img_centers.append(list(center))

        #delete image edges to avoid artifacts created by the undistortion (black at the edges)
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

    closest_img_map = closest_image_map(h_res, w_res, trans_img_centers, trans_img_corners, h, w, downscaling_factor=4,
                                        verbose=True,
                                        margin=1.5)
    # if the resulting image would not fit into memory, we produce a downscaled version
    downscaling_factor = 1
    if (h_res * w_res / downscaling_factor ** 2 > 10000 ** 2):
        downscaling_factor = math.sqrt(h_res * w_res / 10000 ** 2)

    h_res_ds = int(round(h_res / downscaling_factor))
    w_res_ds = int(round(w_res / downscaling_factor))

    closest_img_map = cv2.resize(closest_img_map, (w_res_ds, h_res_ds))
    sigma = 5
    k = 3
    result = [np.zeros((h_res_ds, w_res_ds, 3), dtype='float32')] * (k + 1)
    normalization = [np.zeros((h_res_ds, w_res_ds), dtype='float32')] * (k + 1)

    for i in tqdm(range(0, len(homographys)), desc="combining images", disable=not verbose):

        # add the connected images
        M = np.matmul(offset, homographys[i])
        warped_image = cv2.warpPerspective(list_of_images[i], M,
                                           (w_res, h_res))
        warped_image = cv2.resize(warped_image, (w_res_ds, h_res_ds))

        # multi_band_blending according to http://matthewalunbrown.com/papers/ijcv2007.pdf
        x, y, dx, dy = cv2.boundingRect((closest_img_map == i + 1).astype('uint8'))
        roi = (
            max(0, y - 3 * sigma), min(h_res_ds, y + dy + 3 * sigma), max(0, x - 3 * sigma),
            min(w_res_ds, x + dx + 3 * sigma))
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

    return (end_result).astype('float32')

def homography_from_rotation(yaw,pitch,roll, camera_matrix):
    R_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R_roll = np.array([[math.cos(roll), 0, math.sin(roll)], [0, 1, 0], [-math.sin(roll), 0, math.cos(roll)]])
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R=np.matmul(np.matmul(R_roll,R_pitch),R_yaw)
    RK = np.matmul(R, np.linalg.inv(camera_matrix))
    M = np.matmul(camera_matrix, RK)
    return M

def correct_pose_Homography(yaw_gimbal,pitch_gimbal,roll_gimbal,camera_matrix):
    yaw_gimbal = math.radians(yaw_gimbal)
    pitch_gimbal= math.radians((-90-pitch_gimbal))
    roll_gimbal=math.radians(roll_gimbal)
    M_gimbal=homography_from_rotation(yaw_gimbal,pitch_gimbal,roll_gimbal,camera_matrix)
    return M_gimbal



if __name__ == '__main__':
    # %%

    from random import randrange

    MAX_MATCHES = 10000

    img_dir = r"C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\photos10"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    list_of_lists_of_images = []
    # oneplus 7 pro estimated parameters
    # intrinsic = np.array([[4347.358087366480, 0, 1780.759210199340], [0, 4349.787712956160, 1518.540335312340], [0, 0, 1]])
    # distCoeffs = np.array([0.0928, -0.7394, 0, 0])

    # spark parameters from pix4d
    # intrinsic = np.array([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
    # distCoeffs = np.array([0.117, -0.298, 0.001, 0,0.142])

    # spark parameters from pix4d 2
    intrinsic = np.array([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
    distCoeffs = np.array([0.117, -0.298, 0.001, 0, 0.1420])

    #spark parameters from matlab
    #intrinsic=np.array([[2.83487803e+03, 0.00000000e+00, 2.01766421e+03],[0.00000000e+00, 2.82153143e+03, 1.41937183e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #distCoeffs = np.array([0.29440599, -1.07848387, -0.00455276, 0.00758773, 1.31108063])

    # spark estimated parameters
    # intrinsic = np.array([[3968 * 0.638904348949862, 0, 2048], [0, 2976 * 0.638904348949862, 1536], [0, 0, 1]])
    # distCoeffs = np.array([0.06756436352714615, -0.09146430991012529, 0, 0])
    data = []
    gps_coordinates = []
    xmp_metadata=[]
    data_undistorted = []
    for f1 in files:
        img1 = cv2.imread(f1)
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # data_undistorted.append(cv2.undistort(img1, intrinsic, distCoeffs))
        data.append(img1)
        gps_coord = gpsphoto.getGPSData(f1)
        metadata = pyexiv2.Image(f1)
        xmp_metadata.append(metadata.read_xmp())
        gps_coordinates.append((gps_coord['Latitude'], gps_coord['Longitude']))

    patch_cv2_pickiling()
    #choosing ransac threshold very high to get inliers that don't show up because of distortion (points near the edges)
    features, point_world, homographys_abs, center_image = generate_matrixes(data, gps_coordinates, MAX_MATCHES=MAX_MATCHES,
                                                               sift_threshold=0.6, ransac_threshold=10,
                                                               use_inliers_only=True, goodmatches_threshold=10,purge_multiples=False)
    print('generated matrixes')

    for i in range(len(data)):
        data_undistorted.append(cv2.undistort(data[i], intrinsic, distCoeffs))

    camera_params, points_world_frame, homographys_ba, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations = prepare_data(
        features, point_world, homographys_abs, intrinsic, distCoeffs, verbose=True, outlier_threshold=10)
    x0 = np.hstack((camera_params.ravel(), homographys_ba.ravel(), points_world_frame.ravel()))

    #res_image = multiple_v4(data, homographys_abs, 1, verbose=True)
    #cv2.imwrite('res_image_guess.jpg', res_image)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices,center_image)
    res = least_squares(residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, xtol=1e-8, method='trf',
                        args=(n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame))
    print('performed bundle adjustement')
    plt.plot(res.fun)
    plt.show()
    print('mean residuals= %f' % (np.mean(np.abs(res.fun))))

    intrinsic = res.x[:9].reshape(3, 3)
    distCoeffs = res.x[9:14]
    homographys = res.x[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)

    for i in range(len(data)):
        data_undistorted[i] = cv2.undistort(data[i], intrinsic, distCoeffs)

    #res_image = multiple_v4(data_undistorted, homographys, 1, verbose=True)
    #cv2.imwrite('res_image.jpg', res_image)
    print('intrinsic matrix is:')
    print(intrinsic)
    print('distortion coeffs are:')
    print(distCoeffs)


    #correct for angles
    roll_gimbal=float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalRollDegree'])
    pitch_gimbal=float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalPitchDegree'])
    yaw_gimbal=float(xmp_metadata[center_image]['Xmp.drone-dji.GimbalYawDegree'])

    pose_correction_matrix=correct_pose_Homography(yaw_gimbal,pitch_gimbal,roll_gimbal,intrinsic)
    homograpies_abs_pose_corr=[]
    for M in homographys:
        homograpies_abs_pose_corr.append(np.matmul(pose_correction_matrix,M))
    try:
        res_image = multiple_v4(data_undistorted, homograpies_abs_pose_corr, 1, verbose=True)
        cv2.imwrite('res_image_pose_left.jpg', res_image)
    except:
        print('something went wrong')








