import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shoot_balloon as sb
import pandas as pd
import numpy as np


n_cannons = 3
n_balloons = 3
n_winds = 10
h_min = 0
h_max = 10000
winds = np.linspace(h_min, h_max, num = n_winds + 1)



#cannons = pd.DataFrame({'x': [7100,8000,9000], 'y':[1100, 1150, 1200], 'z' : [50, 25, 130]})
#balloons = pd.DataFrame({'x': [7250, 8100, 9300], 'y':[2300, 2100, 2600], 'z':[1500,1000,2000]})
#ranges = pd.DataFrame({'ran': [4000, 4000, 4000]})


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.setWindowIcon(QIcon('icon.png'))
        
        #initialize data matrix
        self.n_dic = {'cannon_x':n_cannons, 'cannon_y':n_cannons, 'cannon_z':n_cannons, 'balloon_x':n_balloons, 'balloon_y':n_balloons, 'balloon_z':n_balloons}
        self.cannon = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.balloon = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['balloon_x', 'balloon_z', 'balloon_z'])
        self.int_only = QIntValidator(0,99999,self)
        ############################################################
        self.cannon_x_label = QLabel('좌표')    
        self.cannon_z_label = QLabel('높이')
        self.balloon_x_label = QLabel('좌표')
        self.balloon_z_label = QLabel('높이')

        for obj_name in self.n_dic:
            for i in range(self.n_dic[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                #exec('self.{}_{}.textChanged.connect(lambda: self.lineEditChanged("{}_{}", {}, {} ))'.format(obj_name, i, obj_name, i, i, obj_name  ))
        self.cannon_x_0.textChanged.connect(lambda: self.lineEditChanged(self.cannon_x_0, 0, 'cannon_x', self.cannon ))
        ############################################################
        self.height_label = QLabel('고도')
        for i in range(n_winds):
            exec('self.height_label_{} = QLabel("미터")'.format(i))
        
        self.wind_dir_label = QLabel('풍향')
        for i in range(n_winds):
            exec('self.wind_dir_{} = QLineEdit()'.format(i))
        self.wind_vel_label = QLabel('풍속')
        for i in range(n_winds):
            exec('self.wind_vel_{} = QLineEdit()'.format(i))
        ############################################################
        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)






        ##########################################################
        #left layout : 그림이 나오는 곳
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()

        # R1: 발사대
        R1 = QGroupBox('발사대의 좌표와 높이', self)
        
        R1box = QHBoxLayout()

        R1box_1 = QVBoxLayout()
        R1box_1.addWidget(self.cannon_x_label)
        for i in range(n_cannons):
            exec('R1box_1.addWidget(self.cannon_x_{})'.format(i))

        R1box_2 = QVBoxLayout()
        R1box_2.addWidget(self.cannon_x_label)
        for i in range(n_cannons):
            exec('R1box_2.addWidget(self.cannon_y_{})'.format(i))

        R1box_3 = QVBoxLayout()
        R1box_3.addWidget(self.cannon_z_label)
        for i in range(n_cannons):
            exec('R1box_3.addWidget(self.cannon_z_{})'.format(i))
        
        R1box.addLayout(R1box_1)
        R1box.addLayout(R1box_2)
        R1box.addLayout(R1box_3)
        R1.setLayout(R1box)
        
        
        # R2 : 풍선
        R2 = QGroupBox('풍선의 좌표와 높이', self)
        
        R2box = QHBoxLayout()

        R2box_1 = QVBoxLayout()
        R2box_1.addWidget(self.balloon_x_label)
        for i in range(n_balloons):
            exec('R2box_1.addWidget(self.balloon_x_{})'.format(i))
    
        R2box_2 = QVBoxLayout()
        R2box_2.addWidget(self.balloon_z_label)
        for i in range(n_balloons):
            exec('R2box_2.addWidget(self.balloon_z_{})'.format(i))
        
        R2box.addLayout(R2box_1)
        R2box.addLayout(R2box_2)
        R2.setLayout(R2box)
        
      

        # R3 : 바람
        R3 = QGroupBox('고도별 풍향과 풍속', self)
        
        R3box = QHBoxLayout()

        R3box_1 = QVBoxLayout()
        R3box_1.addWidget(self.height_label)
        for i in range(n_winds):
            exec('R3box_1.addWidget(self.height_label_{})'.format(i))
    
        R3box_2 = QVBoxLayout()
        R3box_2.addWidget(self.wind_dir_label)
        for i in range(n_winds):
            exec('R3box_2.addWidget(self.wind_dir_{})'.format(i))
        
        R3box_3 = QVBoxLayout()
        R3box_3.addWidget(self.wind_vel_label)
        for i in range(n_winds):
            exec('R3box_3.addWidget(self.wind_vel_{})'.format(i))
        R3box.addLayout(R3box_1)
        R3box.addLayout(R3box_2)
        R3box.addLayout(R3box_3)
        
        R3.setLayout(R3box)

        # R4 : 버튼
        

        rightLayout.addWidget(R1)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R2)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R3)
        rightLayout.addWidget(self.pushButton)
     
       # 

 
        
        
     
     


        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def pushButtonClicked(self):

        sb.drawplot(10, self.cannons, self.balloons, self.ax)
        img = plt.imread("map.png")
        self.ax.imshow(img, extent=[0, 8000, 0, 5000])   
        self.canvas.draw()




    def lineEditChanged(self, line_name, i, j, mat):
        input_val = line_name.text()
        mat.loc[i,j] = input_val
        print(mat)
      


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()