from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np


class ZoomView(QtWidgets.QLabel):
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
        temp_pos = [e.pos().x(), e.pos().y()]
        temp_pos[0] = int(round(temp_pos[0] * self.scaling_factor - 0.5))
        temp_pos[1] = int(round(temp_pos[1] * self.scaling_factor - 0.5))
        self.position = QtCore.QPoint(temp_pos[0], temp_pos[1])

        self.buttons = e.buttons()
        self.onMouseMoved.emit()

    def resizeEvent(self, event):
        self.onWidgetResized.emit()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self.buttons = e.buttons()
        self.onMousePressed.emit()

    def show_zoomed_image(self, pre_zoom_image, position, zoom_size, pre_zoom_mask):
        x_lower = int(position[0] - (zoom_size / 2))
        x_upper = x_lower + zoom_size
        y_lower = int(position[1] - (zoom_size / 2))
        y_upper = y_lower + zoom_size

        if y_lower < 0:
            y_lower = 0
            y_upper = y_lower + zoom_size

        if y_upper > pre_zoom_image.shape[0]:
            y_upper = pre_zoom_image.shape[0]
            y_lower = y_upper - zoom_size

        if x_lower < 0:
            x_lower = 0
            x_upper = x_lower + zoom_size

        if x_upper > pre_zoom_image.shape[1]:
            x_upper = pre_zoom_image.shape[1]
            x_lower = x_upper - zoom_size

        zoomed_image = pre_zoom_image[y_lower:y_upper, x_lower:x_upper, :]
        zoomed_mask = pre_zoom_mask[y_lower:y_upper, x_lower:x_upper]

        masked_zoom_img = zoomed_image.copy()

        masked_zoom_img[zoomed_mask > 0] = (0, 0, 255)

        masked_zoom_img = cv2.addWeighted(masked_zoom_img, 0.4, zoomed_image, 0.6, 0)

        zoomed_image_converted = cv2.cvtColor(masked_zoom_img, cv2.COLOR_BGR2RGB)

        zoomed_image_converted = QtGui.QImage(zoomed_image_converted, zoomed_image_converted.shape[1],
                                              zoomed_image_converted.shape[0],
                                              zoomed_image_converted.shape[1] * 3,
                                              QtGui.QImage.Format_RGB888)
        zoomed_image_converted = QtGui.QPixmap.fromImage(zoomed_image_converted)

        zoomed_image_converted = zoomed_image_converted.scaled(self.size(), QtCore.Qt.KeepAspectRatio)

        self.scaling_factor = zoomed_image.shape[0] / (zoomed_image_converted.height())

        self.setPixmap(zoomed_image_converted)

        zoom_pos = (x_lower, y_lower)

        return zoom_pos
