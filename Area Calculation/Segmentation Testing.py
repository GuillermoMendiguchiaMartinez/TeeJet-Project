import cv2
import numpy as np

'''This code will compare the ground truth segmentation with 
the segmentation of the user, for a specific piece of land'''

original = cv2.imread(r"C:\Users\lenovo X2500\Aalborg Universitet\Beda Xaver Alexander Berner - Groupexchange\Project\segmentation testing\res_image_oriented_cropped.jpg", cv2.IMREAD_GRAYSCALE) == 255
ground_truth_white = cv2.imread(r"C:\Users\lenovo X2500\Aalborg Universitet\Beda Xaver Alexander Berner - Groupexchange\Project\segmentation testing\Manual segmentation_white.png", cv2.IMREAD_GRAYSCALE) == 255
ground_truth_white=np.logical_and(ground_truth_white, original == 0)
cv2.imwrite("white_segmentation.png", ground_truth_white * 255)

# Here goes the output of the app mask from the GUI
segmentation_compare = cv2.imread(r"C:\Users\lenovo X2500\Aalborg Universitet\Beda Xaver Alexander Berner - Groupexchange\Project\segmentation testing\Test_Kristian_1.PNG", cv2.IMREAD_GRAYSCALE) / 255

ground_truth_area = np.sum(ground_truth_white)
good_segmentation_area = np.sum(np.logical_and(ground_truth_white == 1, segmentation_compare == 1))
bad_segmentation_area = np.sum(np.logical_and(ground_truth_white == 0, segmentation_compare == 1))

print("Ratio of good segmentation is ", good_segmentation_area/ground_truth_area)
print("Ratio of bad segmentation is ", bad_segmentation_area/ground_truth_area)


