import os
import sys
import threading
import subprocess
import math
import matplotlib.pyplot as plt
from Segmentation.SegmentationGuille import *
import cv2

from PyQt5 import QtCore, QtWebEngineWidgets, QtWidgets, uic


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        # start voila server
        self.thread = MyThread()
        self.thread.daemon = True
        self.thread.start()
        # load UI design and show the UI
        uic.loadUi('GUI.ui', self)

        self.image = np.zeros((10, 10, 3)).astype('uint8')
        self.mask = np.zeros((10, 10)).astype('uint8')
        self.mask_downscaled = np.zeros((10, 10)).astype('uint8')
        self.image_downscaled = self.image
        self.position_press = (0, 0)
        self.zoom_slider_value = 505
        self.zoom_position = (0, 0)

        self.show()

        self.SegmentationViewer.onMouseMoved.connect(self.process_mouse_input)
        self.SegmentationViewer.onMousePressed.connect(self.process_mouse_input)
        self.SegmentationViewer.onWidgetResized.connect(self.process_viewer_resize)
        self.hue_threshold_slider.valueChanged.connect(self.process_threshold_slider)
        self.s_v_threshold_slider.valueChanged.connect(self.process_threshold_slider)

        self.ZoomViewer.onMouseMoved.connect(self.process_zoom_mouse_input)
        self.ZoomViewer.onMousePressed.connect(self.process_zoom_mouse_input)
        self.ZoomViewer.onWidgetResized.connect(self.process_zoom_resize)
        self.zoom_slider.valueChanged.connect(self.process_zoom_slider)

        self.export_spray.clicked.connect(self.process_export_spray_pattern)
        self.reset_map.clicked.connect(self.process_reset_map)

    def process_mouse_input(self):
        x = self.SegmentationViewer.position.x()
        y = self.SegmentationViewer.position.y()
        self.x_label.setText('X: %d' % (x))
        self.y_label.setText('Y: %d' % (y))
        if y < self.image.shape[0] and x < self.image.shape[1]:
            self.color = self.image[int(math.floor(y)), int(math.floor(x))]
            self.color_label.setText('Color: ' + (str(self.color)))
            if self.SegmentationViewer.buttons == QtCore.Qt.LeftButton:
                self.position_press = (self.SegmentationViewer.position.x(), self.SegmentationViewer.position.y())
                self.segmentation_color = self.image[int(math.floor(y)), int(math.floor(x))]
                self.chosen_color_label.setText('Chosen Color: ' + (str(self.segmentation_color)))
                self.mask = segment(self.image, self.segmentation_color, (self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
                self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color,
                                               (self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
                self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)
                self.hue_threshold_slider.setEnabled(True)
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)

    def process_zoom_mouse_input(self):
        x_zoom_proportional = self.ZoomViewer.position.x() + self.zoom_position[0]
        y_zoom_proportional = self.ZoomViewer.position.y() + self.zoom_position[1]
        self.x_label.setText('X: %d' % (x_zoom_proportional))
        self.y_label.setText('Y: %d' % (y_zoom_proportional))
        if y_zoom_proportional < self.image.shape[0] and x_zoom_proportional < self.image.shape[1]:
            self.color = self.image[int(math.floor(y_zoom_proportional)), int(math.floor(x_zoom_proportional))]

            self.color_label.setText('Color: ' + (str(self.color)))
            if self.ZoomViewer.buttons == QtCore.Qt.LeftButton:
                self.segmentation_color = self.image[
                    int(math.floor(y_zoom_proportional)), int(math.floor(x_zoom_proportional))]
                self.chosen_color_label.setText('Chosen Color: ' + (str(self.segmentation_color)))
                self.mask = segment(self.image, self.segmentation_color,  (self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
                self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color,
                                    (self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
                self.hue_threshold_slider.setEnabled(True)
                self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                                       self.zoom_slider_value, self.mask)
                self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)

    def process_threshold_slider(self):
        self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color, (self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
        self.mask = segment(self.image, self.segmentation_color,(self.hue_threshold_slider.value(),self.s_v_threshold_slider.value(),self.s_v_threshold_slider.value()))
        self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)
        self.zoom_position = self.ZoomViewer.show_zoomed_image(self.image, self.position_press,
                                                               self.zoom_slider_value, self.mask)

    def closeEvent(self, event):
        # do stuff
        print('closing voila server')
        self.thread.process.kill()
        print(self.thread.is_alive())
        event.accept()  # let the window close

    def load_image(self, image_path, max_dimension=2000):
        self.image = cv2.imread(image_path)
        max_dim_img = max(self.image.shape[0], self.image.shape[1])
        if max_dim_img > max_dimension:
            scaling_factor = max_dimension / max_dim_img
            self.image_downscaled = cv2.resize(self.image, (
                int(round(self.image.shape[1] * scaling_factor)), int(round(self.image.shape[0] * scaling_factor))))
        else:
            self.image_downscaled = self.image
        self.mask = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.mask_downscaled = np.zeros((self.image_downscaled.shape[0], self.image_downscaled.shape[1]))

    def process_viewer_resize(self):
        self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)

    def process_zoom_resize(self):
        if self.SegmentationViewer.position != None:
            self.ZoomViewer.show_zoomed_image(self.image, self.position_press, self.zoom_slider_value,
                                              self.mask)

    def process_zoom_slider(self):
        self.zoom_slider_value = self.zoom_slider.value()
        self.ZoomViewer.show_zoomed_image(self.image, self.position_press, self.zoom_slider_value, self.mask)

    def process_export_spray_pattern(self):
        self.exported_mask = self.mask.copy()
        # To create an upper bound for the used amount of liquid
        spray_resolution = 0.5  # Default: 0.5 m
        image_dim = [3968, 2976]  # Size of a single image from the drone
        height = 25
        cam_fov_deg = 71.5/2

        cam_fov = math.radians(cam_fov_deg)
        width_real = math.tan(cam_fov) * height * 2
        print('width_real: ', width_real)
        pixels_per_m = image_dim[0]/width_real  # How much 1m corresponds to in pixels
        print('Meters per Pixel: ', pixels_per_m)

        kernel_size = int(round(pixels_per_m*spray_resolution))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.exported_mask = cv2.dilate(self.exported_mask, kernel)
        cv2.imwrite("Exported_Mask.png", self.exported_mask)

        white_count = cv2.countNonZero(self.exported_mask)
        m_squared_total = pixels_per_m ** -2 * white_count
        print('Blue area in M^2: ', m_squared_total)
        pesticides_amount = float(self.pesticides_lineedit.text())
        print('Pesticides_Amount: ', pesticides_amount)
        pesticides_estimated_total = pesticides_amount * m_squared_total
        print('Pesticides_Estimated_Total: ', pesticides_estimated_total)
        self.pesticides_label.setText('Estimated Amount of Pesticide: %d L' % (pesticides_estimated_total))


        max_dim_img = max(self.exported_mask.shape[0], self.exported_mask.shape[1])
        max_dimension = 2000
        scaling_factor = max_dimension / max_dim_img
        self.exported_mask = cv2.resize(self.exported_mask, (
            int(round(self.exported_mask.shape[1] * scaling_factor)), int(round(self.exported_mask.shape[0] * scaling_factor))))
        self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape, self.exported_mask)

    def process_reset_map(self):
        window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))



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
