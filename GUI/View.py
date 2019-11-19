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

    def show_image(self, image_downscaled, mask_red, image_org_shape, mask_blue=False):
        masked_img = image_downscaled.copy()
        masked_img[mask_blue > 0] = (255, 0, 0)
        masked_img[mask_red > 0] = (0, 0, 255)
        masked_img = cv2.addWeighted(masked_img, 0.4, image_downscaled, 0.6, 0)
        image_converted = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        image_converted = QtGui.QImage(image_converted, masked_img.shape[1], masked_img.shape[0],
                                       masked_img.shape[1] * 3,
                                       QtGui.QImage.Format_RGB888)
        image_converted = QtGui.QPixmap(image_converted)
        image_converted = image_converted.scaled(self.size(), QtCore.Qt.KeepAspectRatio)

        self.scaling_factor = image_org_shape[0] / (image_converted.height())
        self.setPixmap(image_converted)

