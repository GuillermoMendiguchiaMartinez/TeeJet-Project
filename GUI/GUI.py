import os
import sys
import threading
import subprocess
import math
import matplotlib.pyplot as plt
from Segmentation.SegmentationGuille import *
import cv2

from PyQt5 import QtCore, QtWebEngineWidgets, QtWidgets, uic, QtGui


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        # start voila server
        self.thread = MyThread()
        self.thread.daemon = True
        self.thread.start()
        # load UI design and show the UI
        uic.loadUi('GUI.ui', self)
        self.SegmentationViewer.setAlignment(QtCore.Qt.AlignTop)

        self.image = np.zeros((10, 10, 3)).astype('uint8')
        self.mask = np.zeros((10, 10)).astype('uint8')
        self.mask_downscaled = np.zeros((10, 10)).astype('uint8')
        self.image_downscaled = self.image
        self.position_press = (0, 0)
        self.zoom_slider_value = 505
        self.zoom_position = (0, 0)
        self.zoom_middle_x=False
        self.zoom_middle_y=False
        self.dilated_mask=False
        self.checkBox_show_pesticide = QtWidgets.QCheckBox(self.show_pesticide_checkBox)

        self.show()

        self.SegmentationViewer.onMouseMoved.connect(self.process_mouse_input)
        self.SegmentationViewer.onMousePressed.connect(self.process_mouse_input)
        self.SegmentationViewer.onWidgetResized.connect(self.process_viewer_resize)
        self.hue_threshold_slider.valueChanged.connect(self.process_threshold_slider)
        self.s_v_threshold_slider.valueChanged.connect(self.process_threshold_slider)

        self.ZoomViewer.onMousePressed.connect(self.process_zoom_mouse_input)
        self.ZoomViewer.onWidgetResized.connect(self.process_zoom_resize)
        self.zoom_slider.valueChanged.connect(self.process_zoom_slider)

        self.export_spray.clicked.connect(self.process_export_spray_pattern)
        self.reset_map.clicked.connect(self.process_reset_map)
        self.load_image_btn.clicked.connect(self.process_load_image)
        self.checkBox_show_pesticide.stateChanged.connect(self.update_seg_viewer)

    def process_mouse_input(self):
        x = self.SegmentationViewer.position.x()
        y = self.SegmentationViewer.position.y()

        if y < self.image.shape[0] and x < self.image.shape[1]:
            self.color = self.image[int(math.floor(y)), int(math.floor(x))]
            if self.SegmentationViewer.buttons == QtCore.Qt.LeftButton:
                self.zoom_middle_x = x
                self.zoom_middle_y = y
                self.position_press = (self.SegmentationViewer.position.x(), self.SegmentationViewer.position.y())
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)
                self.update_seg_viewer()

            if self.SegmentationViewer.buttons == QtCore.Qt.RightButton:
                self.segmentation_color = self.image[int(math.floor(y)), int(math.floor(x))]
                self.mask = segment(self.image, self.segmentation_color, (
                self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(),
                self.s_v_threshold_slider.value()))
                self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color,
                                               (self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(),
                                                self.s_v_threshold_slider.value()))
                self.process_calc_dilated_mask()
                self.update_seg_viewer()
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)
                self.hue_threshold_slider.setEnabled(True)
                self.s_v_threshold_slider.setEnabled(True)

    def process_zoom_mouse_input(self):
        x_zoom_proportional = self.ZoomViewer.position.x() + self.zoom_position[0]
        y_zoom_proportional = self.ZoomViewer.position.y() + self.zoom_position[1]
        if y_zoom_proportional < self.image.shape[0] and x_zoom_proportional < self.image.shape[1]:
            self.color = self.image[int(math.floor(y_zoom_proportional)), int(math.floor(x_zoom_proportional))]

            if self.ZoomViewer.buttons == QtCore.Qt.LeftButton:
                self.position_press = (x_zoom_proportional, y_zoom_proportional)
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)
                self.update_seg_viewer()
            if self.ZoomViewer.buttons == QtCore.Qt.RightButton:
                self.segmentation_color = self.image[
                    int(math.floor(y_zoom_proportional)), int(math.floor(x_zoom_proportional))]
                self.mask = segment(self.image, self.segmentation_color, (
                self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(),
                self.s_v_threshold_slider.value()))
                self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color,
                                               (self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(),
                                                self.s_v_threshold_slider.value()))
                self.hue_threshold_slider.setEnabled(True)
                self.s_v_threshold_slider.setEnabled(True)
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)
                self.process_calc_dilated_mask()
                self.update_seg_viewer()

    def process_threshold_slider(self):
        self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color, (
        self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(), self.s_v_threshold_slider.value()))
        self.mask = segment(self.image, self.segmentation_color, (
        self.hue_threshold_slider.value(), self.s_v_threshold_slider.value(), self.s_v_threshold_slider.value()))
        self.process_calc_dilated_mask()
        self.update_seg_viewer()
        self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                               self.zoom_slider_value, self.mask)

    def closeEvent(self, event):
        # do stuff
        print('closing voila server')
        self.thread.process.kill()
        print(self.thread.is_alive())
        event.accept()  # let the window close

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        max_dim_img = max(self.image.shape[0], self.image.shape[1])
        if max_dim_img > 2000:
            self.scaling_factor = 2000 / max_dim_img
            self.image_downscaled = cv2.resize(self.image, (
                int(round(self.image.shape[1] * self.scaling_factor)), int(round(self.image.shape[0] * self.scaling_factor))))
        else:
            self.image_downscaled = self.image
        self.mask = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.mask_downscaled = np.zeros((self.image_downscaled.shape[0], self.image_downscaled.shape[1]))

    def process_viewer_resize(self):
        self.update_seg_viewer()

    def process_zoom_resize(self):
        if self.SegmentationViewer.position != None:
            self.ZoomViewer.show_zoomed_image(self.image, self.position_press, self.zoom_slider_value,
                                              self.mask)

    def process_zoom_slider(self):
        self.zoom_slider_value = self.zoom_slider.value()
        self.ZoomViewer.show_zoomed_image(self.image, self.position_press, self.zoom_slider_value, self.mask)
        self.process_calc_dilated_mask()
        self.update_seg_viewer()

    def process_export_spray_pattern(self):
        save_path = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',filter='PNG(*.PNG)')
        cv2.imwrite(save_path[0], self.mask)

    def process_calc_dilated_mask(self):
        self.dilated_mask = self.mask.copy()
        # To create an upper bound for the used amount of liquid
        spray_resolution = 0.5  # Default: 0.5 m
        image_dim = [3968, 2976]  # Size of a single image from the drone
        height = 25
        cam_fov_deg = 66.55 / 2

        cam_fov = math.radians(cam_fov_deg)
        width_real = math.tan(cam_fov) * height * 2
        pixels_per_m = image_dim[0] / width_real  # How much 1m corresponds to in pixels
        kernel_size = int(round(pixels_per_m * spray_resolution))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.dilated_mask = cv2.dilate(self.dilated_mask, kernel)
        white_count = cv2.countNonZero(self.dilated_mask)
        m_squared_total = pixels_per_m ** -2 * white_count
        pesticides_amount = float(self.pesticides_lineedit.text())
        pesticides_estimated_total = pesticides_amount * m_squared_total
        self.pesticides_label.setText('Estimated Amount of Pesticide: %d L' % pesticides_estimated_total)

        self.dilated_mask = cv2.resize(self.dilated_mask, (
            int(round(self.dilated_mask.shape[1] * self.scaling_factor)),
            int(round(self.dilated_mask.shape[0] * self.scaling_factor))))

    def process_reset_map(self):
        window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))

    def process_load_image(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileName(parent=self, filter="JPG (*.jpg)")[0]
        window.load_image(self.image_path)
        self.update_seg_viewer()
        self.tabWidget.setCurrentIndex(1)

    def update_seg_viewer(self):
        mask_blue=self.dilated_mask*self.checkBox_show_pesticide.isChecked()
        self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape, mask_blue,
                                           zoom_slider_value=self.zoom_slider_value, x=self.zoom_middle_x, y=self.zoom_middle_y,
                                           scaling_factor=self.scaling_factor)

class MyThread(threading.Thread):
    def run(self):
        self.process = subprocess.Popen(['voila', 'mapselect.ipynb', '--port=8866', '--no-browser'])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    path_ui = os.path.join(os.path.dirname(__file__), "Gui.ui")
    window = Ui()
    window.mapWgt.setZoomFactor(1)
    window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))
    window.load_image(r'res_image.jpg')
    window.SegmentationViewer.show_image(window.image_downscaled, window.mask_downscaled, window.image.shape)
    window.show()
    sys.exit(app.exec_())
