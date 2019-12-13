import cv2
import numpy as np

def nothing(x):
    pass

def mouseRGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsB = image[y, x, 0]
        colorsG = image[y, x, 1]
        colorsR = image[y, x, 2]
        colors = image[y, x]
        print("Red: ", colorsR)
        print("Green: ", colorsG)
        print("Blue: ", colorsB)
        print("BGR Format: ", colors)
        print("Coordinates of pixel: X: ", x, "Y: ", y)


def segment(img,color,thresholdwidth,perform_opening=False,perform_closing=False):
    colorHSV=cv2.cvtColor(np.array(color).astype('uint8').reshape(-1,1,3), cv2.COLOR_BGR2HSV)
    lowerBound = np.array([colorHSV[0][0][0]-thresholdwidth[0], max(colorHSV[0][0][1]-thresholdwidth[1],0), max(colorHSV[0][0][2]-thresholdwidth[2],0)])
    #print('LowerBound: ', lowerBound)
    upperBound = np.array([colorHSV[0][0][0]+thresholdwidth[0], min(colorHSV[0][0][1]+thresholdwidth[1],255), min(colorHSV[0][0][2]+thresholdwidth[2],255)])
    #print('Upperbound: ', upperBound)
    if lowerBound[0] < 0:
        lowerBound[0] = 180+lowerBound[0]
    if upperBound[0] > 180:
        upperBound[0] = upperBound[0]-180
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if lowerBound[0] > upperBound[0]:
        mask = cv2.inRange(imgHSV, np.array([0,lowerBound[1],lowerBound[2]]), upperBound)
        mask += cv2.inRange(imgHSV, lowerBound, np.array([180,upperBound[1],upperBound[2]]))
    else:
        mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    if perform_opening:
        kernelOpen = np.ones((5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    if perform_closing:
        kernelClose = np.ones((20, 20))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)
    print('segmentation completed')
    #print(mask.shape)
    return mask



'''
main function starts here
we will call the mouse event and the segmentation event in this part
'''

if __name__ == "__main__":
    print("Please, select the undesired part")
    image = cv2.imread("100m.jpg")
    small2 = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB', mouseRGB)

##############################################################
    # Selection of lower and upper threshold for colour detection
    lowerBound = np.array([36, 25, 25])
    upperBound = np.array([110, 255, 255])

    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))

    # Read image
    img = cv2.imread('100m.jpg', 1)
    img_orig = cv2.imread('100m.jpg', 1)

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)

    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    # Assign the desired mask
    maskFinal = mask

    # Draw the contours
    conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # get second masked value (background) mask must be inverted
    mask_inv = cv2.bitwise_not(maskFinal)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    #recolour the mask to a more visible tone
    bk[mask > 0] = (255, 0, 0)

    # combine foreground+background
    final = cv2.bitwise_or(img, bk)

    small = cv2.resize(final, (0,0), fx=0.25, fy=0.25)

    #cv2.drawContours(small, conts, -1, (255, 0, 0), 1)
    # cv2.imshow("maskClose", maskClose)
    # cv2.imshow("maskOpen", maskOpen)
    # cv2.imshow("mask", mask)
    # cv2.imshow("Original", img_orig)
    cv2.imshow("Segmented", small)

    # Do until esc pressed
    while (1):
        cv2.imshow('mouseRGB', small2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    # if esc pressed, finish.
    cv2.destroyAllWindows()
