import pandas as pd
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sqlite3
import time
import math

from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shoot_balloon as sb

form_class = uic.loadUiType("balloon.ui")[0]

cannons = pd.DataFrame({'x': [7250, 8100, 9300], 'y':[5300, 5100, 5600], 'z':[1500,1000,2000]})
balloons = pd.DataFrame({'x': [7100,8000,9000], 'y':[6100, 6150, 6200], 'z' : [50, 25, 130]})

ranges = pd.DataFrame({'ran': [4000, 4000, 4000]})



class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.qPixmapVar = QPixmap()
        
        #sb.draw_plot(100, cannons, balloons, ranges)
        self.qPixmapVar.load("balloon_map.png")
        self.label_4.setPixmap(self.qPixmapVar)

        
        
       
   
#################################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()