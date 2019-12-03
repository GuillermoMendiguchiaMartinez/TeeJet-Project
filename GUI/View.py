from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np


class View(QtWidgets.QLabel):
    onMouseMoved = QtCore.pyqtSignal()
    onMousePressed = QtCore.pyqtSignal()
    onWidgetResized = QtCore.pyqtSignal()
    scaling_factor: float

    def __init__(self, parent=None):
        self.position = None
        self.scaling_factor = 1
        QtWidgets.QLabel.__init__(self, parent)

    def mouseMoveEvent(self, e) -> None:
        # emit the cursor position when the mouse is moved over the Widget

        self.position = e.pos() * self.scaling_factor
        self.buttons = e.buttons()
        self.onMouseMoved.emit()

    def resizeEvent(self, event):
        self.onWidgetResized.emit()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self.buttons = e.buttons()
        self.onMousePressed.emit()

    def show_image(self, image_downscaled, mask_red, image_org_shape, mask_blue=False, zoom_slider_value=False, x=False, y=False,scaling_factor=False):
        x = x*scaling_factor
        y = y*scaling_factor
        zoom_slider_value=int(zoom_slider_value*scaling_factor/2)
        masked_img = image_downscaled.copy()
        masked_img[mask_blue > 0] = (255, 0, 0)
        masked_img[mask_red > 0] = (0, 0, 255)
        masked_img = cv2.addWeighted(masked_img, 0.4, image_downscaled, 0.6, 0)

        x_lower = int(x - zoom_slider_value)
        x_upper = int(x + zoom_slider_value)
        y_lower = int(y - zoom_slider_value)
        y_upper = int(y + zoom_slider_value)

        if y_lower < 0:
            y_lower = 0
            y_upper = y_lower + zoom_slider_value*2

        if y_upper > masked_img.shape[0]:
            y_upper = masked_img.shape[0]
            y_lower = y_upper - zoom_slider_value*2

        if x_lower < 0:
            x_lower = 0
            x_upper = x_lower + zoom_slider_value*2

        if x_upper > masked_img.shape[1]:
            x_upper = masked_img.shape[1]
            x_lower = x_upper - zoom_slider_value*2

        start_point=(x_lower, y_lower)
        end_point=(x_upper, y_upper)

        masked_img=cv2.rectangle(masked_img, start_point, end_point, (90, 90, 90), 3)

        image_converted = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        image_converted = QtGui.QImage(image_converted, masked_img.shape[1], masked_img.shape[0],
                                       masked_img.shape[1] * 3,
                                       QtGui.QImage.Format_RGB888)
        image_converted = QtGui.QPixmap(image_converted)
        image_converted = image_converted.scaled(self.size(), QtCore.Qt.KeepAspectRatio)

        self.scaling_factor = image_org_shape[0] / (image_converted.height())
        self.setPixmap(image_converted)

