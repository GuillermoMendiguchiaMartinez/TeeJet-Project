from PyQt5 import uic, QtWidgets, QtCore
import sys

class Ui(QtWidgets.QMainWindow):

    def __init__(self):

        super(Ui,self).__init__()

        uic.loadUi("TestGui.ui",self)

        self.printBtn.clicked.connect(self.printmsg)
        self.show()

    def printmsg(self):
        self.outputLbl.setText("Hallo World")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())

