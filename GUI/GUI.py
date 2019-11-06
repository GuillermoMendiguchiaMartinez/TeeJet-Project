import os
import sys
import threading
import subprocess
import math
import matplotlib.pyplot as plt
from Segmentation.SegmentationGuille import *

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

        self.show()

        self.SegmentationViewer.onMouseMoved.connect(self.process_mouse_input)
        self.SegmentationViewer.onWidgetResized.connect(self.process_viewer_resize)

        self.SegmentationViewer.onMousePressed.connect(self.process_mouse_input)
        self.threshold_slider.valueChanged.connect(self.process_threshold_slider)

    def process_mouse_input(self):
        x = self.SegmentationViewer.position.x()
        y = self.SegmentationViewer.position.y()
        self.x_label.setText('x: %d' % (x))
        self.y_label.setText('y: %d' % (y))
        if y < self.image.shape[0] and x < self.image.shape[1]:
            self.color = self.image[int(math.floor(y)), int(math.floor(x))]

            self.color_label.setText('color: ' + (str(self.color)))

            if self.SegmentationViewer.buttons == QtCore.Qt.LeftButton:
                self.segmentation_color = self.image[int(math.floor(y)), int(math.floor(x))]
                self.chosen_color_label.setText('chosen color: ' + (str(self.segmentation_color)))
                self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color,
                                               self.threshold_slider.value())
                self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)
                self.threshold_slider.setEnabled(True)

    def process_threshold_slider(self):
        self.mask_downscaled = segment(self.image_downscaled, self.segmentation_color, self.threshold_slider.value())
        self.SegmentationViewer.show_image(self.image_downscaled, self.mask_downscaled, self.image.shape)

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


class MyThread(threading.Thread):
    def run(self):
        self.process = subprocess.Popen(['voila', 'mapselect.ipynb', '--port=8866', '--no-browser'])


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    path_ui = os.path.join(os.path.dirname(__file__), "Gui.ui")
    window = Ui()
    window.mapWgt.setZoomFactor(1)
    window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))

    window.load_image(r'C:\Users\bedab\OneDrive\AAU\TeeJet-Project\res_image.jpg')
    window.SegmentationViewer.show_image(window.image_downscaled, window.mask_downscaled, window.image.shape)
    window.show()
    sys.exit(app.exec_())
