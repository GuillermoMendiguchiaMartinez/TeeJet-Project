from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np


class SprayView(QtWidgets.QLabel):
    def __init__(self, parent=None):
        QtWidgets.QLabel.__init__(self, parent)