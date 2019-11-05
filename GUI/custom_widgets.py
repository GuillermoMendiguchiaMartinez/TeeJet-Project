from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np


class View(QtWidgets.QLabel):
    onMouseMoved = QtCore.pyqtSignal()
    onMousePressed = QtCore.pyqtSignal()
    resized = QtCore.pyqtSignal()
    scaling_factor: float

    def __init__(self, parent=None):
        self.image=np.zeros((10,10,3)).astype('uint8')
        #self.image=cv2.imread(r'C:\Users\bedab\OneDrive\AAU\TeeJet-Project\Stitching\demonstration pictures\center picture 02\res_image.jpg')
        self.mask=np.zeros((10,10)).astype('uint8')
        self.mask_downscaled=np.zeros((10,10)).astype('uint8')
        self.image_org=self.image
        QtWidgets.QLabel.__init__(self, parent)
        self.scaling_factor=1


    def mouseMoveEvent(self, e) -> None:
        # emit the cursor position when the mouse is moved over the Widget

        self.position = e.pos() * self.scaling_factor
        self.buttons = e.buttons()
        self.onMouseMoved.emit()

    def resizeEvent(self, event):
        self.show_image()
        self.resized.emit()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self.buttons = e.buttons()
        self.onMousePressed.emit()


    def load_image(self, image_path,max_dimension=2000):
        self.image_org=cv2.imread(image_path)
        max_dim_img=max(self.image_org.shape[0],self.image_org.shape[1])
        if max_dim_img>max_dimension:
            scaling_factor=max_dimension/max_dim_img
            self.image=cv2.resize(self.image_org,(int(round(self.image_org.shape[1]*scaling_factor)),int(round(self.image_org.shape[0]*scaling_factor))))
        else:
            self.image=self.image_org
        self.mask = np.zeros((self.image_org.shape[0],self.image_org.shape[1]))
        self.mask_downscaled = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.setMouseTracking(True)



    def show_image(self):
        masked_img=self.image.copy()
        masked_img[self.mask_downscaled>0]=(0,0,255)
        image_org = cv2.addWeighted(masked_img, 0.4, self.image, 0.6, 0)
        image_converted=cv2.cvtColor(image_org,cv2.COLOR_RGB2BGR)

        image_converted = QtGui.QImage(image_converted, image_org.shape[1], image_org.shape[0], image_org.shape[1] * 3,
                                       QtGui.QImage.Format_RGB888)
        image_converted = QtGui.QPixmap(image_converted)
        image_converted=image_converted.scaled(self.size(),QtCore.Qt.KeepAspectRatio)

        self.scaling_factor = self.image_org.shape[0] / (image_converted.height())
        print((image_converted.height(),image_converted.width()))
        self.setPixmap(image_converted)
