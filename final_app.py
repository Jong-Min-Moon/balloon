import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shoot_balloon as sb
import pandas as pd
import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
from scipy.optimize import fmin_cobyla

n_cannons = 2
n_balloons = 2
n_enemys = 2
n_winds = 5
h_min = 0
h_max = 7000
winds = np.linspace(h_min, h_max, num = n_winds + 1)
winds_idx = zip(winds[:-1].astype(int), winds[1:].astype(int))



#cannons = pd.DataFrame({'x': [7100,8000,9000], 'y':[1100, 1150, 1200], 'z' : [50, 25, 130]})
#balloons = pd.DataFrame({'x': [7250, 8100, 9300], 'y':[2300, 2100, 2600], 'z':[1500,1000,2000]})
ranges = pd.DataFrame({'ran': [5000, 5000]})


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.setWindowIcon(QIcon('icon.png'))
        
        #initialize data matrix
        self.wind_dir = pd.DataFrame({
            'wind_dir': [1,2,3,4,5,6,7,8],
            'vec':[ [0,1], [np.sqrt(0.5), np.sqrt(0.5)],
                    [1,0], [np.sqrt(0.5), -np.sqrt(0.5)],
                    [0,-1], [-np.sqrt(0.5), -np.sqrt(0.5)],
                    [-1,0], [-np.sqrt(0.5), np.sqrt(0.5)]]})
        
        self.n_dic = {'cannon_x':n_cannons, 'cannon_y':n_cannons, 'cannon_z':n_cannons, 'balloon_x':n_balloons, 'balloon_y':n_balloons, 'balloon_z':n_balloons,  'thetav_v1': n_balloons, 'thetav_v2':n_balloons}
        self.n_dic2 = {'cannon2_x':n_cannons, 'cannon2_y':n_cannons, 'enemy_x':n_enemys, 'enemy_y':n_enemys}

        self.data = {}
        self.data['cannon'] = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.data['cannon'].iloc[0,0] = 550
        self.data['cannon'].iloc[0,1] = 200
        self.data['cannon'].iloc[0,2] = 200
        self.data['cannon'].iloc[1,0] = 1500
        self.data['cannon'].iloc[1,1] = 400
        self.data['cannon'].iloc[1,2] = 300

        self.data['balloon'] = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['balloon_x', 'balloon_y', 'balloon_z'])
        self.data['balloon'].iloc[0,0] = 700
        self.data['balloon'].iloc[0,1] = 700
        self.data['balloon'].iloc[0,2] = 1000
        self.data['balloon'].iloc[1,0] = 1600
        self.data['balloon'].iloc[1,1] = 900
        self.data['balloon'].iloc[1,2] = 1200
        
        self.data['thetav'] = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['thetav_theta', 'thetav_v1', 'thetav_v2'])


        self.data['wind'] = pd.DataFrame(np.zeros((n_winds, 2)), index =winds_idx,  columns = ['wind_dir', 'wind_vel'])
        self.data['wind'].iloc[0,0] = 1
        self.data['wind'].iloc[0,1] = 1
        self.data['wind'].iloc[1,0] = 2
        self.data['wind'].iloc[1,1] = 2
        self.data['wind'].iloc[2,0] = 1
        self.data['wind'].iloc[2,1] = 1
        self.data['wind'].iloc[3,0] = 2
        self.data['wind'].iloc[3,1] = 2
        self.data['wind'].iloc[4,0] = 1
        self.data['wind'].iloc[4,1] = 1
        

        self.data['cannon2'] = pd.DataFrame(np.zeros((n_cannons, 2)), columns = ['cannon2_x', 'cannon2_y'])
        self.data['cannon2'].iloc[0,0] = 550
        self.data['cannon2'].iloc[0,1] = 200
        self.data['cannon2'].iloc[1,0] = 1500
        self.data['cannon2'].iloc[1,1] = 400
   
        self.data['enemy'] = pd.DataFrame(np.zeros((n_enemys, 2)), columns = ['enemy_x', 'enemy_y'])
        self.data['enemy'].iloc[0,0] = 700
        self.data['enemy'].iloc[0,1] = 700
        self.data['enemy'].iloc[1,0] = 1600
        self.data['enemy'].iloc[1,1] = 900

        #Validators
        self.wind_dir_only = QIntValidator(1, 8)
        self.int_only = QIntValidator(0,99999)
        self.theta = QDoubleValidator(0, 1.57079, 5)
        self.v= QDoubleValidator(-999, 999, 5)
        
        #initialize plt plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.OptimOutput = QTextBrowser(self)
       
        
        #대포와 풍선
       

        self.cannon_balloon_list = {}
        for obj_name in self.n_dic:
            for i in range(self.n_dic[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                exec('self.cannon_balloon_list["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        
        self.theta_list = {}
        for i in range(n_balloons):
            exec('self.{}_{} = QLineEdit()'.format('thetav_theta', i))
            exec('self.{}_{}.setValidator(self.theta)'.format('thetav_theta', i))
            exec('self.theta_list["self.{}_{}"] = self.{}_{}'.format('thetav_theta', i, 'thetav_theta', i))

        self.v_list = {}
        for i in range(n_balloons):
            exec('self.{}_{} = QLineEdit()'.format('thetav_v1', i))
            exec('self.{}_{}.setValidator(self.v)'.format('thetav_v1', i))
            exec('self.v_list["self.{}_{}"] = self.{}_{}'.format('thetav_v1', i, 'thetav_v1', i))


        for i in range(n_balloons):
            exec('self.{}_{} = QLineEdit()'.format('thetav_v2', i))
            exec('self.{}_{}.setValidator(self.v)'.format('thetav_v2', i))
            exec('self.v_list["self.{}_{}"] = self.{}_{}'.format('thetav_v2', i, 'thetav_v2', i))
        
        for widget in self.cannon_balloon_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.cannon_balloon_list[widget].textChanged.connect(lambda a = widget, this_widget = self.cannon_balloon_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged(this_widget, this_i, this_j, this_mat ))

        for widget in self.theta_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.theta_list[widget].textChanged.connect(lambda a = widget, this_widget = self.theta_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged4(this_widget, this_i, this_j, this_mat ))         
        
        for widget in self.v_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.v_list[widget].textChanged.connect(lambda a = widget, this_widget = self.v_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged4(this_widget, this_i, this_j, this_mat ))         
        
        #대포 발사각
        for i in range(n_cannons):
            exec('self.cannon_theta_{} = QLabel("  °")'.format(i))

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

        #대포 - 풍선 매칭
        for i in range(n_cannons):
            exec('self.cannon_alloc_{} = QLabel("포 {}")'.format(i, i+1))
        for i in range(n_balloons):
            exec('self.alloc_{} = QLabel("-----")'.format(i))
            exec('self.balloon_alloc_{} = QLabel("풍선 ")'.format(i))
            exec('self.idland_{} = QLabel("")'.format(i))
            exec('self.actland_{} = QLabel("")'.format(i))

        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        self.thetaButton = QPushButton("차트그리기(각)")
        self.thetaButton.clicked.connect(self.thetaButtonClicked)
        
        ####################################################################################
        #GUI 2
        self.cannon_enemy_list = {}
        for obj_name in self.n_dic2:
            for i in range(self.n_dic2[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                exec('self.cannon_enemy_list["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for widget in self.cannon_enemy_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.cannon_enemy_list[widget].textChanged.connect(lambda a = widget, this_widget = self.cannon_enemy_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged3(this_widget, this_i, this_j, this_mat ))
        
        #대포 발사각
        for i in range(n_cannons):
            exec('self.cannon2_theta_{} = QLabel("  °")'.format(i))

        self.pushButton2 = QPushButton("발사각 계산하기(최적화 알고리즘)")
        self.pushButton2.clicked.connect(self.OptimAlgo)

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

        R_G1_box_G3 = QGroupBox('발사각', self)
        R_G1_box_G3_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G3_box.addWidget(self.cannon_theta_{})'.format(i))
        R_G1_box_G3.setLayout(R_G1_box_G3_box)

        R_G1_box.addWidget(R_G1_box_G1)
        R_G1_box.addWidget(R_G1_box_G2)
        R_G1_box.addWidget(R_G1_box_G3)
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
        

        #R3 : 바람
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

        # R_G4: 발사 결과
        R_G4 = QGroupBox('발사 결과', self)  
        R_G4_box = QHBoxLayout()

        R_G4_box_G1 = QGroupBox('포-풍선 매칭', self)
        R_G4_box_G1_box = QHBoxLayout()
        R_G4_box_G1_box_1 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G1_box_1.addWidget(self.cannon_alloc_{})'.format(i))
        R_G4_box_G1_box_2 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G1_box_2.addWidget(self.alloc_{})'.format(i))
        R_G4_box_G1_box_3 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G1_box_3.addWidget(self.balloon_alloc_{})'.format(i))

        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_1)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_2)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_3)
        R_G4_box_G1.setLayout(R_G4_box_G1_box)

        R_G4_box_G2 = QGroupBox('이론적 탄착점', self)
        R_G4_box_G2_box = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G2_box.addWidget(self.idland_{})'.format(i))
        R_G4_box_G2.setLayout(R_G4_box_G2_box)

        R_G4_box_G3 = QGroupBox('실제 탄착군의 중심점', self)
        R_G4_box_G3_box = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G3_box.addWidget(self.actland_{})'.format(i))
        R_G4_box_G3.setLayout(R_G4_box_G3_box)

        R_G4_box.addWidget(R_G4_box_G1)
        R_G4_box.addWidget(R_G4_box_G2)
        R_G4_box.addWidget(R_G4_box_G3)
        R_G4.setLayout(R_G4_box)

        






        R_G5 = QGroupBox('발사각과 발사방향', self)  
        R_G5_box = QHBoxLayout()

        R_G5_box_G1 = QGroupBox('발사각', self)
        R_G5_box_G1_box = QHBoxLayout()
        R_G5_box_G1_box_1 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G5_box_G1_box_1.addWidget(self.thetav_v1_{})'.format(i))
        R_G5_box_G1_box_2 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G5_box_G1_box_2.addWidget(self.thetav_v2_{})'.format(i))
        R_G5_box_G1_box.addLayout(R_G5_box_G1_box_1)
        R_G5_box_G1_box.addLayout(R_G5_box_G1_box_2)
        R_G5_box_G1.setLayout(R_G5_box_G1_box)

        R_G5_box_G2 = QGroupBox('발사방향', self)
        R_G5_box_G2_box = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G5_box_G2_box.addWidget(self.thetav_theta_{})'.format(i))
        R_G5_box_G2.setLayout(R_G5_box_G2_box)

        R_G5_box.addWidget(R_G5_box_G2)
        R_G5_box.addWidget(R_G5_box_G1)
        R_G5.setLayout(R_G5_box)


        rightLayout.addWidget(R_G1)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G2)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R3)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G4)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G5)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addWidget(self.thetaButton)
##########################################################################

        rightLayout2 = QVBoxLayout()
        # R_G1: 발사대
        R2_G1 = QGroupBox('발사대', self)  
        R2_G1_box = QHBoxLayout()

        R2_G1_box_G1 = QGroupBox('좌표', self)
        R2_G1_box_G1_box = QHBoxLayout()
        R2_G1_box_G1_box_1 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R2_G1_box_G1_box_1.addWidget(self.cannon2_x_{})'.format(i))
        R2_G1_box_G1_box_2 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R2_G1_box_G1_box_2.addWidget(self.cannon2_y_{})'.format(i))
        R2_G1_box_G1_box.addLayout(R2_G1_box_G1_box_1)
        R2_G1_box_G1_box.addLayout(R2_G1_box_G1_box_2)
        R2_G1_box_G1.setLayout(R2_G1_box_G1_box)

   

        R2_G1_box_G3 = QGroupBox('발사각', self)
        R2_G1_box_G3_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('R2_G1_box_G3_box.addWidget(self.cannon2_theta_{})'.format(i))
        R2_G1_box_G3.setLayout(R2_G1_box_G3_box)

        R2_G1_box.addWidget(R2_G1_box_G1)

        R2_G1_box.addWidget(R2_G1_box_G3)
        R2_G1.setLayout(R2_G1_box)


    # R_G2: 적
        R2_G2 = QGroupBox('적', self)  
        R2_G2_box = QHBoxLayout()

        R2_G2_box_G1 = QGroupBox('좌표', self)
        R2_G2_box_G1_box = QHBoxLayout()
        R2_G2_box_G1_box_1 = QVBoxLayout()
        for i in range(n_enemys):
            exec('R2_G2_box_G1_box_1.addWidget(self.enemy_x_{})'.format(i))
        R2_G2_box_G1_box_2 = QVBoxLayout()
        for i in range(n_enemys):
            exec('R2_G2_box_G1_box_2.addWidget(self.enemy_y_{})'.format(i))
        R2_G2_box_G1_box.addLayout(R2_G2_box_G1_box_1)
        R2_G2_box_G1_box.addLayout(R2_G2_box_G1_box_2)
        R2_G2_box_G1.setLayout(R2_G2_box_G1_box)

        R2_G2_box.addWidget(R2_G2_box_G1)
  
        R2_G2.setLayout(R2_G2_box)

        rightLayout2.addWidget(R2_G1)
        rightLayout2.addStretch(3)
        rightLayout2.addWidget(R2_G2)
        rightLayout2.addStretch(3)
        rightLayout2.addStretch(3)
        rightLayout2.addWidget(self.OptimOutput)
        rightLayout2.addWidget(self.pushButton2)
        
##########################################################################

     
       # 

 
        
        
     
     


        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.addLayout(rightLayout2)
        
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        layout.setStretchFactor(rightLayout2, 0)
        self.setLayout(layout)

    def pushButtonClicked(self):
        if sum(self.data['cannon'].cannon_z >= self.data['balloon'].balloon_z ) > 0:
            QMessageBox.about(self, "오류", "포의 고도가 풍선의 고도보다 높거나 같으면 발사할 수 없습니다")
        else:
            self.ax.clear()
            per, idland, actland = sb.drawplot(100, self.data['cannon'], self.data['balloon'], self.data['wind'], self.ax, ranges)
            print('idland:', idland)
            print('actland:', actland)
            for i in range(len(per)):
                exec('self.balloon_alloc_{}.setText("풍선 {}")'.format(i, per[i]+1))
                exec('self.idland_{}.setText("{}")'.format(i, idland[i].round(2)))
                exec('self.actland_{}.setText("{}")'.format(i, actland[i].round(2)))
            img = plt.imread("map.png")
            limit = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
            self.ax.imshow(img, extent=limit)   
            self.canvas.draw()

    def thetaButtonClicked(self):
        if sum(self.data['cannon'].cannon_z >= self.data['balloon'].balloon_z ) > 0:
            QMessageBox.about(self, "오류", "포의 고도가 풍선의 고도보다 높거나 같으면 발사할 수 없습니다")
        else:
            self.ax.clear()
            per, idland, actland = sb.drawplot(100, self.data['cannon'], self.data['balloon'], self.data['wind'], self.ax, ranges)
            print('idland:', idland)
            print('actland:', actland)
            for i in range(len(per)):
                exec('self.balloon_alloc_{}.setText("풍선 {}")'.format(i, per[i]+1))
                exec('self.idland_{}.setText("{}")'.format(i, idland[i].round(2)))
                exec('self.actland_{}.setText("{}")'.format(i, actland[i].round(2)))
            img = plt.imread("map.png")
            limit = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
            self.ax.imshow(img, extent=limit)   
            self.canvas.draw()
    
    def OptimAlgo(self):
        wind_tbl = pd.merge(self.data['wind'], self.wind_dir, how = 'left', on = 'wind_dir').set_index(self.data['wind'].index)
        cannon = np.array(self.data['cannon2'].iloc[0, :])
        cannon_0 = np.append(cannon, 0)
        enemy = np.array(self.data['enemy'].iloc[0, :])
        print('cannon, enemy:', cannon, enemy)
        ran = 5000
        under = la.norm(enemy - cannon)
        print('밑변:', under )
        if ran <= under:
            QMessageBox.about(self, "오류", "포와 적의 거리가 너무 멀어서({}) 현재 사거리로는 목표지점에 착탄 불가합니다.".format(under))
        else:
            theta_init = np.arccos(under / ran )
            print('theta_init:', theta_init)
            direc_init = (enemy - cannon) / la.norm(enemy - cannon)
            print('direc_init:', direc_init)
                
            def f(x):
                return (sb.optim(100, x[0], np.array([x[1],x[2]]), cannon_0, enemy, wind_tbl, ran))
            def constr1(x):
                x[1]
            def constr2(x):
                np.pi/2 - x[1]
            def constr3(x):
                x[2]
            def constr4(x):
                np.pi/2 - x[2]
            minimum = fmin_cobyla(f, [theta_init, direc_init[0], direc_init[1]], [constr1, constr2, constr3, constr4], rhoend=1e-7)
            opt_eval = (sb.shoot_for_optim(100, cannon_0, sb.ang2coord(cannon[:1], minimum[0], minimum[1:], 5000), wind_tbl, 5000))
            self.OptimOutput.setPlainText(str(minimum) + ' ' + str(opt_eval))
                #  : 3000})
            
                # self.OptimOutput.setPlainText(str(minimum) + '\n 최적화 해의 값을 가지고 100발 쐈을 때의 탄착군 중심: {}'.format(opt_eval))



    def lineEditChanged(self, widget, i, j, mat):
        try:
            mat.loc[i,j] = int(widget.text())
            cannon_vec = np.array(self.data['cannon'].iloc[i,:])
            balloon_vec = np.array(self.data['balloon'].iloc[i,:])
            print(cannon_vec, balloon_vec)
            _,_,_, th, _ = sb.peak_xy(cannon_vec, balloon_vec, 5000)
            theta = str(round((np.pi / 2 - th) * (180 / np.pi),4))
            exec('self.cannon_theta_{}.setText(theta + "°")'.format(i))
        except:
            print('입력이 없음')
        print(mat)
      
    def lineEditChanged2(self, widget, i, j, mat):
        try:
            mat.iloc[i,j] = int(widget.text())
        except:
            print('입력이 없음')
        print(mat) 

    def lineEditChanged3(self, widget, i, j, mat):
        try:
            mat.loc[i,j] = int(widget.text())
        except:
            print('입력이 없음')
        print(mat)

    def lineEditChanged4(self, widget, i, j, mat):
        try:
            mat.loc[i,j] = float(widget.text())
        except:
            print('입력이 없음')
        print(mat)
if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()