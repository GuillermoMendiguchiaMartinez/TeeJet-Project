from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np

class ZoomView(QtWidgets.QLabel):

    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)

    def show_zoomed_image(self, pre_zoom_image, position, dim):
        y_lower = int(position.y()-(dim/2))
        y_upper = y_lower+dim
        x_lower = int(position.x()-(dim/2))
        x_upper = x_lower+dim

        if y_lower < 0:
            y_lower = 0
            y_upper = y_lower+dim

        if y_upper > pre_zoom_image.shape[0]:
            y_upper = pre_zoom_image.shape[0]
            y_lower = y_upper-dim

        if x_lower < 0:
            x_lower = 0
            x_upper = x_lower + dim

        if x_upper > pre_zoom_image.shape[1]:
            x_upper = pre_zoom_image.shape[1]
            x_lower = x_upper - dim

        zoomed_image = pre_zoom_image[y_lower:y_upper,x_lower:x_upper,:]

        zoomed_image_converted = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)

        zoomed_image_converted = QtGui.QImage(zoomed_image_converted, zoomed_image_converted.shape[1], zoomed_image_converted.shape[0],
                                       zoomed_image_converted.shape[1] * 3,
                                       QtGui.QImage.Format_RGB888)
        zoomed_image_converted = QtGui.QPixmap.fromImage(zoomed_image_converted)

        self.setPixmap(zoomed_image_converted)
