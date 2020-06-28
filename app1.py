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
        self.n_dic = {'cannon_x':n_cannons, 'cannon_y':n_cannons, 'cannon_z':n_cannons, 'balloon_x':n_balloons, 'balloon_y':n_balloons, 'balloon_z':n_balloons}
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
        


        self.wind_dir_only = QIntValidator(1, 8, self)
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
            self.cannon_balloon_list[widget].textChanged.connect(lambda a = widget, this_widget = self.cannon_balloon_list[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged(this_widget, this_i, this_j, this_mat ))
          
        
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

        

        rightLayout.addWidget(R_G1)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G2)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R3)
        rightLayout.addStretch(3)
        rightLayout.addWidget(R_G4)
        rightLayout.addWidget(self.pushButton)
     
       # 

 
        
        
     
     


        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

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


if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()