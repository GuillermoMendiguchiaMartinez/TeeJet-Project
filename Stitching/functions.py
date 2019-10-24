import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sys
import glob
import os


def homography_from_rotation(angle, camera_matrix, axis=0):
    if axis == 0:
        R = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
    elif axis == 1:
        R = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    elif axis == 2:
        R = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    RK = np.matmul(R, np.linalg.inv(camera_matrix))
    M = np.matmul(camera_matrix, RK)
    return M


def generate_matrixes(list_of_images, w, MAX_MATCHES=1000):
    sift = cv2.xfeatures2d.SIFT_create(MAX_MATCHES)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    h = math.ceil(len(list_of_images) / w)
    idx_matrix = np.arange((h * w))
    idx_matrix = idx_matrix.reshape(h, w)
    if idx_matrix.shape[0] != h:
        idx_matrix = np.transpose(idx_matrix)
    print(idx_matrix.shape)
    idx_matrix[idx_matrix >= len(list_of_images)] = -255
    features = {}
    for i in range(h):
        for j in range(w):
            idx = idx_matrix[i, j]
            if idx != -255:
                img = cv2.cvtColor(list_of_images[idx], cv2.COLOR_RGB2GRAY)

                features[(i, j)] = sift.detectAndCompute(img, None)
    matches = {}
    for i in range(h - 1):
        for j in range(max(w - 1, 1)):
            if idx_matrix[i, j] >= 0 and idx_matrix[i + 1, j] >= 0:
                all_matches = flann.knnMatch(features[(i, j)][1], features[(i + 1, j)][1], k=2)
                goodmatches = []
                for m, n in all_matches:
                    if m.distance < 0.7 * n.distance:
                        goodmatches.append(m)
                if len(goodmatches)<4:
                    print('not enough matches found!!!')

                print(i)
                print(j)
                print(len(goodmatches))
                matches[((i, j), (i + 1, j))] = goodmatches
            if w != 1:
                if idx_matrix[i, j] >= 0 and idx_matrix[i, j + 1] >= 0:
                    all_matches = flann.knnMatch(features[i, j][1], features[i, j + 1][1], k=2)
                    goodmatches = []
                    for m, n in all_matches:
                        if m.distance < 0.7 * n.distance:
                            goodmatches.append(m)
                    matches[((i, j), (i, j + 1))] = goodmatches

    return idx_matrix, features, matches


def find_homography(img1, img2, MIN_MATCH_COUNT=4, MAX_MATCHES=10000):
    '''
    finds Homography for 2 given Images
    :param img1: image that will be transformed
    :param img2: image that stays
    :param MIN_MATCH_COUNT: min ammount of matches that are needed, otherwise an error is shown
    :param MAX_MATCHES: max ammount of matches that will be extracted in each picture
    :return: Homography Matrix M
    '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(MAX_MATCHES)
    # brisk = cv2.BRISK_create(60)
    # orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    print(len(keypoints1))
    print(len(keypoints2))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print(len(good))

    # matches = sorted(matches, key = lambda x:x.distance)
    # good=matches[0:1000]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    return M


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
    result = np.round(cv2.perspectiveTransform(a, M))
    return result.reshape(1, 2)[0].astype('int')


def multiple_v2(list_of_lists_of_images, connections, homographys, og_coordinate, margin=1):
    '''
    takes images and stitches them together
    :param list_of_lists_of_images: images in the format [[0,1,...,w],[0,1,...w],...,[0,1,...v]] v<w
    :param connections: how the pictures are connected
    :param homographys: homographys according to the connections
    :param og_coordinate: coordinates of the center image (will not be transformed)
    :return: stitched image
    '''
    # move the center image into the middle
    # offset_y = og_coordinate[0] * list_of_lists_of_images[0][0].shape[0]
    # offset_x = og_coordinate[0] * list_of_lists_of_images[0][0].shape[1]
    w = list_of_lists_of_images[0][0].shape[1]
    h = list_of_lists_of_images[0][0].shape[0]

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
    trans_img_centers.append([0.5 * w, 0.5 * h])

    for i in range(len(trans_img_centers)):
        trans_img_centers[i][1] += offset_y
        trans_img_centers[i][0] += offset_x

    offset = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])

    closest_img_map = closest_image_map(h_res, w_res, trans_img_centers, h, w, downscaling_factor=4, verbose=True, )
    # generate empty
    result = np.zeros((h_res, w_res, 3), dtype='uint8')

    for i in range(0, len(homographys)):
        # add the connected images
        M = np.matmul(offset, homographys[i])
        warped_image = cv2.warpPerspective(list_of_lists_of_images[connections[i][0][0]][connections[i][0][1]], M,
                                           (result.shape[1], result.shape[0]))
        warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
        result[result == 0] = warped_image[result == 0]
        result[np.logical_and(closest_img_map == i + 1, warped_gray != 0)] = warped_image[
            np.logical_and(closest_img_map == i + 1, warped_gray != 0)]

    # add the center img
    center_img = np.zeros((h_res, w_res, 3), dtype='uint8')

    center_img[offset_y:offset_y + h, offset_x:offset_x + w, :] = list_of_lists_of_images[og_coordinate[0]][
        og_coordinate[1]]
    center_gray = cv2.cvtColor(center_img, cv2.COLOR_RGB2GRAY)
    result[result == 0] = center_img[result == 0]
    result[np.logical_and(closest_img_map == i + 2, center_gray != 0)] = center_img[
        np.logical_and(closest_img_map == i + 2, center_gray != 0)]

    return result


if __name__ == '__main__':
    # %%

    from random import randrange

    MAX_MATCHES = 30000

    img_dir = r"C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\test"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    list_of_lists_of_images = []
    # parameters read out from opendronemap
    intrinsic = np.array([[3968 * 0.638904348949862, 0, 2048], [0, 2976 * 0.638904348949862, 1536], [0, 0, 1]])
    distCoeffs = np.array([0.06756436352714615, -0.09146430991012529, 0, 0])

    for f1 in files:
        img1 = cv2.imread(f1)
        cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.undistort(img1, intrinsic, distCoeffs)[200:-200, 200:-200, :]
        data.append(img1)

    w = 1
    t = len(data)
    center_coordinates = (3, 0)

    idx, features, matches = generate_matrixes(data, w)
    print('generated matrixes')

    min_x_rot_angle = 0
    min_x_rot = 100
    for angle in np.arange(-3, 3, 0.01):
        center_angle_matrix = homography_from_rotation(angle, intrinsic, 0)

        warped_features = features.copy()
        keypoints = features[center_coordinates][0]
        warped_keypoints = cv2.perspectiveTransform(cv2.KeyPoint_convert(keypoints).reshape(-1, 1, 2),
                                                    center_angle_matrix).astype('float')
        warped_keypoints = list(
            map(tuple, warped_keypoints.reshape((warped_keypoints.shape[0], warped_keypoints.shape[2]))))

        warped_matches = matches.copy()

        for i in range(len(warped_keypoints)):
            warped_keypoints[i] = cv2.KeyPoint_convert([warped_keypoints[i]])[0]
        warped_features[center_coordinates] = (warped_keypoints, features[center_coordinates][1])

        M = np.eye(3)

        for y in range(center_coordinates[0], 1, -1):
            keypoints1 = warped_features[(y, center_coordinates[1])][0]
            keypoints2 = warped_features[(y - 1, center_coordinates[1])][0]
            match = (matches[((y - 1, center_coordinates[1]), (y, center_coordinates[1]))])

            src_pts = np.float32([keypoints2[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

            M_iter, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M_iter = np.linalg.inv(M_iter)
            M_iter, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M = np.matmul(M,M_iter)

        decomposed = cv2.decomposeHomographyMat(M, intrinsic)
        est_angle, _ = cv2.Rodrigues(decomposed[1][0])
        print(angle)
        print(est_angle)
        if abs(est_angle[0]%(math.pi)) < min_x_rot:
            min_x_rot = abs(est_angle[0]%(math.pi))
            min_x_rot_angle = angle
    print('best angle is:')
    print(min_x_rot_angle)

    # print(features)

    """for i in range(0, t // w):
        row = data[i * w:(i + 1) * w]
        list_of_lists_of_images.append(row)

    if t % w != 0:
        i = i + 1
        row = data[i * w:-1]
        list_of_lists_of_images.append(row)

    og_coordinate = (4, 0)
    homographys = []
    connections = []
    # strategy go vertical first
    m_og = og_coordinate[0]
    n_og = og_coordinate[1]
    m = m_og

    vertical_index = 0
    while (m > 0):
        m -= 1

        homography = find_homography(list_of_lists_of_images[m][n_og], list_of_lists_of_images[m + 1][n_og])
        if m + 1 < m_og:
            homography = np.matmul(homographys[-1], homography)
        homographys.append(homography)
        connections.append(((m, n_og), (m_og, n_og)))
    m = m_og
    while m < len(list_of_lists_of_images) - 1:
        m += 1
        homography = find_homography(list_of_lists_of_images[m][n_og], list_of_lists_of_images[m - 1][n_og])
        if m - 1 > m_og:
            homography = np.matmul(homographys[-1], homography)
        homographys.append(homography)
        connections.append(((m, n_og), (m_og, n_og)))
    print(homographys)
    print(connections)

    # homographys = [find_homography(list_of_lists_of_images[0][1], list_of_lists_of_images[1][1]),
    #              find_homography(list_of_lists_of_images[2][1], list_of_lists_of_images[1][1])]
    # print(homographys)
    # connections = [((0, 1), (1, 1)), ((2, 1), (1, 1))]

    res = multiple_v2(list_of_lists_of_images, connections, homographys, og_coordinate)

    fig = plt.figure(figsize=(18, 16), dpi=200, facecolor='w', edgecolor='k')
    cv2.imwrite('res.jpg', res)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()"""
