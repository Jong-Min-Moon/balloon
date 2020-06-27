import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shoot_balloon as sb
import pandas as pd
import numpy as np


n_cannons = 2
n_balloons = 2
n_winds = 5
h_min = 0
h_max = 10000
winds = np.linspace(h_min, h_max, num = n_winds + 1)
winds_idx = zip(winds[:-1].astype(int), winds[1:].astype(int))



#cannons = pd.DataFrame({'x': [7100,8000,9000], 'y':[1100, 1150, 1200], 'z' : [50, 25, 130]})
#balloons = pd.DataFrame({'x': [7250, 8100, 9300], 'y':[2300, 2100, 2600], 'z':[1500,1000,2000]})
ranges = pd.DataFrame({'ran': [4000, 4000, 4000]})


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
        self.data = {}
        self.data['cannon'] = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.data['balloon'] = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['balloon_x', 'balloon_y', 'balloon_z'])
        self.data['wind'] = pd.DataFrame(np.zeros((n_winds, 2)), index =winds_idx,  columns = ['wind_dir', 'wind_vel'])
        
        self.wind_dir_only = QIntValidator(0, 7, self)
        self.int_only = QIntValidator(0,99999,self)
        
        
        #initialize plt plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
       
        
        #대포와 풍선
       
        self.balloon_x_label = QLabel('좌표')
        self.balloon_z_label = QLabel('높이')

        self.cannon_balloon_list = {}
        for obj_name in self.n_dic:
            for i in range(self.n_dic[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                exec('self.cannon_balloon_list["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for widget in self.cannon_balloon_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            print(i, j)
            self.cannon_balloon_list[widget].textChanged.connect(lambda a = widget, this_widget = self.cannon_balloon_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged(this_widget, this_i, this_j, this_mat ))
          
        
        

        #바람
        
        self.height_label = QLabel('고도(m)')
        for i , idx in enumerate(self.data['wind'].index):
            exec('self.height_label_{} = QLabel("{}-{}")'.format(i, idx[0], idx[1]))
        
        self.wind_dir_list = {}
        self.wind_dir_label = QLabel('풍향')
        for i in range(n_winds):
            exec('self.wind_dir_{} = QLineEdit()'.format(i))
            exec("self.wind_dir_list['self.wind_dir_{}'] = self.wind_dir_{}".format(i,i))
        
        for widget in self.wind_dir_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            mat = self.data[inputs[0]]
            self.wind_dir_list[widget].textChanged.connect(lambda a = widget, this_widget = self.wind_dir_list[widget], this_i = i,  this_mat = mat: self.lineEditChanged2(this_widget, this_i, 0, this_mat ))
          

        self.wind_vel_list = {}
        self.wind_vel_label = QLabel('풍속')
        for i in range(n_winds):
            exec('self.wind_vel_{} = QLineEdit()'.format(i))
            exec("self.wind_vel_list['self.wind_vel_{}'] = self.wind_vel_{}".format(i,i))



        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        

        






        ##########################################################
        #left layout : 그림이 나오는 곳
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()

        # R_G1: 발사대
        R_G1 = QGroupBox('발사대', self)  
        R_G1_box = QHBoxLayout()

        R_G1_box_G1 = QGroupBox('좌표', self)
        R_G1_box_G1_box = QHBoxLayout()
        R_G1_box_G1_box_1 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G1_box_1.addWidget(self.cannon_x_{})'.format(i))
        R_G1_box_G1_box_2 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G1_box_2.addWidget(self.cannon_y_{})'.format(i))
        R_G1_box_G1_box.addLayout(R_G1_box_G1_box_1)
        R_G1_box_G1_box.addLayout(R_G1_box_G1_box_2)
        R_G1_box_G1.setLayout(R_G1_box_G1_box)

        R_G1_box_G2 = QGroupBox('높이', self)
        R_G1_box_G2_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G2_box.addWidget(self.cannon_z_{})'.format(i))
        R_G1_box_G2.setLayout(R_G1_box_G2_box)

        R_G1_box.addWidget(R_G1_box_G1)
        R_G1_box.addWidget(R_G1_box_G2)
        R_G1.setLayout(R_G1_box)


        # R_G2: 풍선
        R_G2 = QGroupBox('풍선', self)  
        R_G2_box = QHBoxLayout()

        R_G2_box_G1 = QGroupBox('좌표', self)
        R_G2_box_G1_box = QHBoxLayout()
        R_G2_box_G1_box_1 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G1_box_1.addWidget(self.balloon_x_{})'.format(i))
        R_G2_box_G1_box_2 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G1_box_2.addWidget(self.balloon_y_{})'.format(i))
        R_G2_box_G1_box.addLayout(R_G2_box_G1_box_1)
        R_G2_box_G1_box.addLayout(R_G2_box_G1_box_2)
        R_G2_box_G1.setLayout(R_G2_box_G1_box)

        R_G2_box_G2 = QGroupBox('높이', self)
        R_G2_box_G2_box = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G2_box.addWidget(self.balloon_z_{})'.format(i))
        R_G2_box_G2.setLayout(R_G2_box_G2_box)

        R_G2_box.addWidget(R_G2_box_G1)
        R_G2_box.addWidget(R_G2_box_G2)
        R_G2.setLayout(R_G2_box)
        

          # R_G3: 바람
        # R_G3 = QGroupBox('고도별 풍향과 풍속', self)  
        # R_G3_box = QHBoxLayout()

        # R_G3_box_G1 = QGroupBox('풍향', self)
        # R_G3_box_G1_box = QVBoxLayout()
        # for i in range(n_winds):
        #     exec('R_G3_box_G1_box.addWidget(self.wind_dir_{})'.format(i))
        # R_G3_box_G1.setLayout(R_G3_box_G1_box)

        # R_G3_box_G2 = QGroupBox('풍속', self)
        # R_G3_box_G2_box = QVBoxLayout()
        # for i in range(n_winds):
        #     exec('R_G3_box_G2_box.addWidget(self.wind_vel_{})'.format(i))
        # R_G3_box_G2.setLayout(R_G3_box_G2_box)

        # R_G3_box.addWidget(R_G3_box_G1)
        # R_G3_box.addWidget(R_G3_box_G2)
        # R_G3.setLayout(R_G3_box)
        

        R3 : 바람
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
            exec('self.wind_dir_{}.setValidator(self.wind_dir_only)'.format(i))

        
        R3box_3 = QVBoxLayout()
        R3box_3.addWidget(self.wind_vel_label)
        for i in range(n_winds):
            exec('R3box_3.addWidget(self.wind_vel_{})'.format(i))
            exec('self.wind_vel_{}.setValidator(self.int_only)'.format(i))
        R3box.addLayout(R3box_1)
        R3box.addLayout(R3box_2)
        R3box.addLayout(R3box_3)
        
        R3.setLayout(R3box)

        # R4 : 버튼
        

        rightLayout.addWidget(R_G1)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G2)
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

        sb.drawplot(10, self.data['cannon'], self.data['balloon'], self.data['wind'], self.ax)
        img = plt.imread("map.png")
        self.ax.imshow(img, extent=[0, 8000, 0, 5000])   
        self.canvas.draw()




    def lineEditChanged(self, widget, i, j, mat):
        try:
            mat.loc[i,j] = int(widget.text())
        except:
            print('입력이 없음')
        print(mat)
      
    def lineEditChanged2(self, widget, i, j, mat):
        try:
            mat.iloc[i,j] = int(widget.text())
        except:
            print('입력이 없음')
        print(mat) 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()