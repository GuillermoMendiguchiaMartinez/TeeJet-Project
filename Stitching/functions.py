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


def match_and_extract_points(position, target, sift_threshold,
                             matrix_idx,
                             ransac_threshold,
                             use_inliers_only,
                             flann, features, matches,
                             homographys_rel,
                             point_world_idx,
                             point_world):
    if matrix_idx[position[0], position[1]] != -255 and matrix_idx[target[0], target[1]] != -255:
        all_matches = flann.knnMatch(features[position][1], features[target][1], k=2)
        goodmatches = []
        for m, n in all_matches:
            if m.distance < sift_threshold * n.distance:
                goodmatches.append(m)
        if len(goodmatches) < 4:
            print('not enough matches found!!!')

        print(position)
        print(target)
        print(len(goodmatches))

        keypoints1, _ = features[position]
        keypoints2, _ = features[target]
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in goodmatches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in goodmatches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        homographys_rel[(position, target)] = M

        if use_inliers_only:
            goodmatches = np.array(goodmatches)
            goodmatches = goodmatches.reshape(mask.shape)
            matches[(position, target)] = goodmatches[mask == 1]
        else:
            matches[(position, target)] = goodmatches

        # add the points to the world_points
        for match in matches[(position, target)]:
            query = (matrix_idx[position[0], position[1]], match.queryIdx, position)
            train = (matrix_idx[target[0], target[1]], match.trainIdx, target)
            if query not in point_world_idx:
                if train not in point_world_idx:
                    # creates new point_world if neither keypoint is found in another point_world
                    point_world.append(
                        [(0, 0), set([query, train]),
                         set([matrix_idx[position[0], position[1]], matrix_idx[target[0], target[1]]]), True])
                    point_world_idx[query] = len(point_world) - 1
                    point_world_idx[train] = len(point_world) - 1
                else:
                    # if train is found in a point_world, query is added to that point
                    point_world[point_world_idx[train]][1].add(query)
                    if matrix_idx[position[0], position[1]] in point_world[point_world_idx[train]][2]:
                        print('Warning 2 points of same image added to same world point')
                        print('query')
                        print(point_world_idx[train])
                        point_world[point_world_idx[train]][3]=False
                    point_world[point_world_idx[train]][2].add(matrix_idx[position[0], position[1]])
                    point_world_idx[query] = point_world_idx[train]
            else:
                if train not in point_world_idx:
                    # if query is found in a point_world,train is added to that point
                    point_world[point_world_idx[query]][1].add(train)
                    if matrix_idx[target[0], target[1]] in point_world[point_world_idx[query]][2]:
                        print('Warning 2 points of same image added to same world point')
                        print('train')
                        print(point_world_idx[query])
                        point_world[point_world_idx[query]][3]=False
                    point_world[point_world_idx[query]][2].add(matrix_idx[target[0], target[1]])
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
                            point_world[point_world_idx[query]][3]=False

                        point_world[point_world_idx[query]][2] = point_world[point_world_idx[query]][
                            2].union(point_world[point_world_idx[train]][2])
                        for point in point_world[point_world_idx[train]][1]:
                            point_world_idx[point] = point_world_idx[query]
                        point_world[point_world_idx[train]][3] = False

    return matches, homographys_rel, point_world_idx, point_world


def homography_path_iter(position, target, matrix_idx, homographys_rel, homography=np.eye(3)):
    if position[0] < target[0] and matrix_idx[position[0] + 1, position[1]] != -255:
        M = homographys_rel[position, (position[0] + 1, position[1])]
        M = np.linalg.inv(M)
        homography = np.matmul(homography, M)
        result = homography_path_iter((position[0] + 1, position[1]), target, matrix_idx=matrix_idx,
                                      homographys_rel=homographys_rel, homography=homography)
    elif position[1] < target[1] and matrix_idx[position[0], position[1] + 1] != -255:
        M = homographys_rel[position, (position[0], position[1] + 1)]
        M = np.linalg.inv(M)
        homography = np.matmul(homography, M)
        result = homography_path_iter((position[0], position[1] + 1), target, matrix_idx=matrix_idx,
                                      homographys_rel=homographys_rel, homography=homography)
    elif position[0] > target[0] and matrix_idx[position[0] - 1, position[1]] != 255:
        M = homographys_rel[(position[0] - 1, position[1]), position]
        homography = np.matmul(homography, M)
        result = homography_path_iter((position[0] - 1, position[1]), target, matrix_idx=matrix_idx,
                                      homographys_rel=homographys_rel, homography=homography)
    elif position[1] > target[1]:
        M = homographys_rel[(position[0], position[1] - 1), position]
        homography = np.matmul(homography, M)
        result = homography_path_iter((position[0], position[1] - 1), target, matrix_idx=matrix_idx,
                                      homographys_rel=homographys_rel, homography=homography)
    else:
        result = homography
    return result


def generate_matrixes(list_of_images, w, MAX_MATCHES=5000, center_image=(0, 0), use_inliers_only=True,
                      ransac_threshold=10, sift_threshold=0.7,verbose=True):
    sift = cv2.xfeatures2d.SIFT_create(MAX_MATCHES)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    h = math.ceil(len(list_of_images) / w)

    matrix_idx = np.arange((h * w))
    matrix_idx = matrix_idx.reshape(h, w)
    if matrix_idx.shape[0] != h:
        matrix_idx = np.transpose(matrix_idx)
    print(matrix_idx.shape)
    matrix_idx[matrix_idx >= len(list_of_images)] = -255
    if verbose:
        print('start feature extraction')
    features = {}
    with tqdm(total=h*w, desc="feature extraction",disable=not verbose) as tExtraction:
        for i in range(h):
            for j in range(w):
                idx = matrix_idx[i, j]
                if idx != -255:
                    img = cv2.cvtColor(list_of_images[idx], cv2.COLOR_RGB2GRAY)

                    features[(i, j)] = sift.detectAndCompute(img, None)
                tExtraction.update()

    matches = {}
    homographys_rel = {}
    point_world_idx = {}
    point_world = []
    if verbose:
        print('start matching')
    with tqdm(total=h * w, desc="matching", disable=not verbose) as tMatching:
        for i in range(h):
            for j in range(w):
                if i < h - 1:

                    if matrix_idx[i, j] >= 0 and matrix_idx[i + 1, j] >= 0:
                        matches, homographys_rel, point_world_idx, point_world = match_and_extract_points((i, j),
                                                                                                          (i + 1, j),
                                                                                                          sift_threshold,
                                                                                                          matrix_idx,
                                                                                                          ransac_threshold,
                                                                                                          use_inliers_only,
                                                                                                          flann, features,
                                                                                                          matches,
                                                                                                          homographys_rel,
                                                                                                          point_world_idx,
                                                                                                          point_world)


                if j < w - 1:
                    if matrix_idx[i, j] >= 0 and matrix_idx[i, j + 1] >= 0:
                        matches, homographys_rel, point_world_idx, point_world = match_and_extract_points((i, j),
                                                                                                      (i, j + 1),
                                                                                                      sift_threshold,
                                                                                                      matrix_idx,
                                                                                                      ransac_threshold,
                                                                                                      use_inliers_only,
                                                                                                      flann, features,
                                                                                                      matches,
                                                                                                      homographys_rel,
                                                                                                      point_world_idx,
                                                                                                      point_world)
                tMatching.update()

    # calculate the homographys from the center image to each image
    homographys_abs = [None] * matrix_idx[matrix_idx != -255].size
    for i in range(h):
        for j in range(w):
            if matrix_idx[(i, j)] != -255:
                homographys_abs[matrix_idx[i, j]] = homography_path_iter(center_image, (i, j), matrix_idx,
                                                                         homographys_rel)
    point_world_usable=[]
    for point in point_world:
        if point[3]:
            point_world_usable.append(point)

    point_world=point_world_usable
    # calculate an estimate for each point_world
    for i in range(len(point_world)):
        point_idx = list(point_world[i][1])[0]
        est_homography = homographys_abs[point_idx[0]]
        start_coord = features[point_idx[2]][0][point_idx[1]].pt
        est_world_pos = apply_homography_to_point(start_coord, est_homography)
        point_world[i][0] = est_world_pos


    return matrix_idx, features, matches, point_world, point_world_idx, homographys_abs


def prepare_data(matrix_idx, features, matches, point_world, point_world_idx, homographys_abs, intrinsic, dist_coeffs):

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
            points_camera_frame.append(features[point_idx[j][2]][0][point_idx[j][1]].pt)

        points_world_frame[2 * i:2 * i + 2] = point_world[i][0].astype(float)
    n_observations = len(points_camera_frame)
    homographys = np.empty(n_cameras * 9)
    for i in range(n_cameras):
        homographys[9 * i:9 * i + 9] = homographys_abs[i].ravel()

    points_camera_frame = np.array(points_camera_frame)
    n = 9 * n_cameras + 2 * n_points + 13
    m = 2 * points_camera_frame.shape[0]
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    return camera_params, points_world_frame, homographys, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations


def residuals(params, n_cameras, n_points, n_observations, camera_indicies, point_indicies, points_camera_frame):
    intrinsic = params[:9].reshape(3, 3)
    undistort_coeffs = params[9:14]
    homographys = params[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)
    points_world_frame = params[14 + n_cameras * 9:].reshape(n_points, 2)

    residuals = np.empty(n_observations * 2)
    for i in range(n_observations):
        point_distorted = points_camera_frame[i].astype('float32')
        point_undistorted = cv2.undistortPoints(point_distorted.reshape(-1, 1, 2), intrinsic, undistort_coeffs,
                                                R=np.eye(3), P=intrinsic)
        point_warped = apply_homography_to_point(point_undistorted, homographys[camera_indicies[i]])
        residuals[2 * i:2 * i + 2] = (point_warped - points_world_frame[point_indicies[i]]).reshape(2)
    return residuals


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    m = camera_indices.size * 2
    n = 14 + n_cameras * 9 + n_points * 2
    A = lil_matrix((m, n), dtype=int)
    # camera intrinsics are same for all observations
    A[:, :14] = 1

    """
    for i in range(len(camera_indices)):
        # camera homographys
        A[2*i,14+camera_indices[i]*9:14+camera_indices[i]*9+9]=1
        A[2 * i+1, 14 + camera_indices[i] * 9:14 + camera_indices[i] * 9 + 9] = 1
        # world points
        A[2 * i, 14 + n_cameras * 9 + point_indices[i] * 2]=1
        A[2 * i+1, 14 + n_cameras * 9 + point_indices[i] * 2] = 1
        """
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, 14+camera_indices * 9 + s] = 1
        A[2 * i + 1,14+ camera_indices * 9 + s] = 1
    for s in range(2):
        A[2 * i, 14+n_cameras * 9 + point_indices * 2 + s] = 1
        A[2 * i + 1, 14+n_cameras * 9 + point_indices * 2 + s] = 1
    fig = plt.figure()
    plt.spy(A)
    plt.show()
    fig.savefig('sparse.png', dpi=500)
    return A


def closest_image_map(height_ori, width_ori, coordinates, h_img, w_img, downscaling_factor=4, margin=1,
                      verbose=False):
    '''
    generates image in which the closest origin image to pixels is indicated
    :param height_ori: height of the stitched output
    :param width_ori: width of the stitched output
    :param coordinates: (transformed) centerpoints of the images
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
    new_coordinates = []
    for coordinate in coordinates:
        new_coordinates.append((round(coordinate[1] / downscaling_factor), round(coordinate[0] / downscaling_factor)))
    coordinates = new_coordinates
    h_img = round(h_img / downscaling_factor)
    w_img = round(w_img / downscaling_factor)

    a = np.zeros((height, width))
    x = np.arange(0, a.shape[0], 1)
    y = np.arange(0, a.shape[1], 1)
    xx, yy = np.meshgrid(y, x)
    shortest_distance = np.full_like(a, math.sqrt(height ** 2 + width ** 2), dtype=np.single)
    node = np.zeros_like(shortest_distance, dtype=np.uint8)
    if verbose:
        print(time.time() - start)

    for i in range(0, len(coordinates)):
        x_coord = coordinates[i][1]
        y_coord = coordinates[i][0]

        y_lower = int(max(0, round(y_coord - (h_img * margin))))
        y_upper = int(round(y_coord + (h_img * margin)))
        x_lower = int(max(0, round(x_coord - (w_img * margin))))
        x_upper = int(round(x_coord + (w_img * margin)))
        shortest_distance_roi = shortest_distance[y_lower:y_upper, x_lower:x_upper]
        node_roi = node[y_lower:y_upper, x_lower:x_upper]
        xx_roi = xx[y_lower:y_upper, x_lower:x_upper]
        yy_roi = yy[y_lower:y_upper, x_lower:x_upper]
        shortest_distance_2_roi = np.sqrt((yy_roi - coordinates[i][0]) ** 2 + (xx_roi - coordinates[i][1]) ** 2)

        shorter_matrix = shortest_distance_roi > shortest_distance_2_roi
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

    trans_img_centers.append([0.5 * w, 0.5 * h])

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


if __name__ == '__main__':
    # %%

    from random import randrange

    MAX_MATCHES = 20000

    img_dir = r"C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\photos4"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    list_of_lists_of_images = []
    # oneplus 7 pro estimated parameters
    #intrinsic = np.array([[4347.358087366480, 0, 1780.759210199340], [0, 4349.787712956160, 1518.540335312340], [0, 0, 1]])
    #distCoeffs = np.array([0.0928, -0.7394, 0, 0])


    #spark parameters from pix4d
    #intrinsic = np.array([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
    #distCoeffs = np.array([0.117, -0.298, 0.001, 0,0.142])
    #spark parameters from pix4d 2
    intrinsic = np.array([[2951, 0, 1976], [0, 2951, 1474], [0, 0, 1]])
    distCoeffs = np.array([0.117, -0.298, 0.001, 0,0.1420])
    #spark estimated parameters
    #intrinsic = np.array([[3968 * 0.638904348949862, 0, 2048], [0, 2976 * 0.638904348949862, 1536], [0, 0, 1]])
    #distCoeffs = np.array([0.06756436352714615, -0.09146430991012529, 0, 0])
    data = []
    data_undistorted=[]
    for f1 in files:
        img1 = cv2.imread(f1)
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        data_undistorted.append(cv2.undistort(img1, intrinsic, distCoeffs))
        data.append(img1)

    w = 5
    t = len(data)
    center_coordinates = (0, 2)

    idx, features, matches, points_world, point_world_idx, homographys = generate_matrixes(data, w,MAX_MATCHES=MAX_MATCHES,
                                                                                           center_image=center_coordinates,
                                                                                           sift_threshold=0.7,ransac_threshold=5,use_inliers_only=True)
    print('generated matrixes')

    camera_params, points_world_frame, homographys_ba, camera_indices, point_indices, points_camera_frame, n_cameras, n_points, n_observations = prepare_data(
        idx, features,
        matches,
        points_world,
        point_world_idx,
        homographys,
        intrinsic,
        distCoeffs)
    x0 = np.hstack((camera_params.ravel(), homographys_ba.ravel(), points_world_frame.ravel()))
    print('made initial guess')
    f0 = residuals(x0, n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame)
    plt.plot(f0)
    plt.show()

    res_image = multiple_v3(data_undistorted, homographys, 1)
    plt.imshow(res_image)
    cv2.imwrite('res_image_guess.jpg', res_image)


    print('mean residuals= %f' % (np.mean(np.abs(f0))))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4,xtol=1e-12, method='trf',
                        args=(n_cameras, n_points, n_observations, camera_indices, point_indices, points_camera_frame))
    print('performed bundle adjustement')
    plt.plot(res.fun)
    plt.show()
    print('mean residuals= %f'%(np.mean(np.abs(res.fun))))

    intrinsic = res.x[:9].reshape(3, 3)
    distCoeffs = res.x[9:14]
    homographys = res.x[14:14 + n_cameras * 9].reshape(n_cameras, 3, 3)

    for i in range(len(data)):
        data[i]=cv2.undistort(data[i], intrinsic, distCoeffs)

    res_image = multiple_v3(data, homographys, 1)
    plt.imshow(res_image)
    cv2.imwrite('res_image.jpg', res_image)
    plt.show()
    print('intrinsic matrix is:')
    print(intrinsic)
    print('distortion coeffs are:')
    print (distCoeffs)
