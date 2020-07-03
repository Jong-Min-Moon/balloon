# 적 고사포가 풍선 사격시 바람의 영향을 고려하여 낙탄지점을 예측하는 GUI 프로그램



import pandas as pd #데이터프레임 이용하기 위한 패키지
import numpy as np #행렬 및 벡터 연산을 위한 패키지
import matplotlib.pyplot as plt #그래프 작성을 위한 패키지

#GUI 앱을 위한 패키지
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QRegExp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas #GUI앱에 matplotlib으로 그래프를 그리기 위한 패키지


import shoot_balloon as sb #낙탄지점 예측 알고리즘이 들어있는 코드

#상황 설정
n_cannons = 5 #적 고사포의 개수
n_balloons = 3 #풍선의 개수
winds = np.linspace(0, 8000, num = 12 + 1) #고도 0미터부터 8000미터까지의 구간을 아래와 같이 12개 구간으로 나눔
winds_idx = [ (0,200), (201,500), (501,1000), (1001, 1500), (1501, 2000), (2001,2500), (2501,3000), (3001,4000), (4001,5000), (5001,6000), (6001,7000), (7001,8000) ]



class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
    
    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("적 고사포 낙탄 예측")
        self.setWindowIcon(QIcon('icon.png'))
        
        def setQlineEditLength(QLE): 
            fm = QLE.fontMetrics()
            m = QLE.textMargins()
            c = QLE.contentsMargins()
            w = 6.2*fm.width('x')+m.left()+m.right()+c.left()+c.right()
            QLE.setMaximumWidth(w+8)
        
        e = QLineEdit()

        setQlineEditLength(e)
        layout = QHBoxLayout()
        layout.addWidget(e)   
        
        

        
   


        self.setLayout(layout)


        






if __name__ == "__main__":

    

    app = QApplication(sys.argv)
    app.setFont(QFont("NanumGothic", 14))
    window = MyWindow()
    window.show()
    app.exec_()