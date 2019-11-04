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
        #start voila server
        self.thread = MyThread()
        self.thread.daemon = True
        self.thread.start()
        # load UI design and show the UI
        uic.loadUi('GUI.ui', self)
        self.show()

        self.SegmentationViewer.onMouseMoved.connect(self.process_mouse_input)
        self.SegmentationViewer.onMousePressed.connect(self.process_mouse_input)
        self.threshold_slider.valueChanged.connect(self.process_threshold_slider)
    def process_mouse_input(self):
        x=self.SegmentationViewer.position.x()
        y=self.SegmentationViewer.position.y()
        self.x_label.setText('x: %d'%(x))
        self.y_label.setText('y: %d' % (y))
        if y<self.SegmentationViewer.image_org.shape[0] and x<self.SegmentationViewer.image_org.shape[1]:
            self.color=self.SegmentationViewer.image_org[int(math.floor(y)),int(math.floor(x))]

            self.color_label.setText('color: '+(str(self.color)))

            if self.SegmentationViewer.buttons == QtCore.Qt.LeftButton:
                self.segmentation_color=self.SegmentationViewer.image_org[int(math.floor(y)),int(math.floor(x))]
                self.chosen_color_label.setText('chosen color: '+(str(self.segmentation_color)))

                self.SegmentationViewer.mask_downscaled=segment(self.SegmentationViewer.image,self.segmentation_color,self.threshold_slider.value())
                self.SegmentationViewer.show_image()
                self.threshold_slider.setEnabled(True)
    def process_threshold_slider(self):
        self.SegmentationViewer.mask_downscaled = segment(self.SegmentationViewer.image, self.segmentation_color,
                                                          self.threshold_slider.value())
        self.SegmentationViewer.show_image()


    def closeEvent(self, event):
        # do stuff
        print('closing voila server')
        self.thread.process.kill()
        print(self.thread.is_alive())
        event.accept()  # let the window close



class MyThread(threading.Thread):
    def run(self):
        self.process = subprocess.Popen(['voila', 'mapselect.ipynb', '--port=8866','--no-browser'])



if __name__ == '__main__':


    app = QtWidgets.QApplication(sys.argv)
    path_ui = os.path.join(os.path.dirname(__file__), "Gui.ui")
    window = Ui()
    window.mapWgt.setZoomFactor(1)
    window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))
    window.show()
    window.SegmentationViewer.load_image(r'C:\Users\bedab\OneDrive\AAU\TeeJet-Project\res_image.jpg')
    window.SegmentationViewer.show_image()
    sys.exit(app.exec_())
