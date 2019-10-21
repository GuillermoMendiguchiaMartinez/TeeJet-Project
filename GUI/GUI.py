import os
import sys
import threading

from PyQt5 import QtCore, QtWebEngineWidgets, QtWidgets, uic

if __name__ == '__main__':
    class MyThread(threading.Thread):
        def run(self):
            os.system("voila mapselect.ipynb --no-browser")
            pass

    thread = MyThread()
    thread.daemon = True
    thread.start()

    app = QtWidgets.QApplication(sys.argv)
    path_ui = os.path.join(os.path.dirname(__file__), "Gui.ui")
    window = uic.loadUi(path_ui)
    window.mapWgt.setZoomFactor(1)
    window.mapWgt.load(QtCore.QUrl('http://localhost:8866'))
    window.show()
    sys.exit(app.exec_())
