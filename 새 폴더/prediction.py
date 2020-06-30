import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import shoot_balloon as sb
import predict_angle as pa
import pandas as pd
import numpy as np
from numpy import linalg as la
from scipy.optimize import fmin_cobyla

n_cannons = 5
n_balloons = 5
n_winds = 12
h_min = 0
h_max = 7000
winds = np.linspace(h_min, h_max, num = n_winds + 1)
winds_idx = [ (0,200), (201,500), (501,1000), (1001, 1500), (1501, 2000), (2001,2500), (2501,3000), (3001,4000), (4001,5000), (5001,6000), (6001,7000), (7001,8000) ]



#cannons = pd.DataFrame({'x': [7100,8000,9000], 'y':[1100, 1150, 1200], 'z' : [50, 25, 130]})
#balloons = pd.DataFrame({'x': [7250, 8100, 9300], 'y':[2300, 2100, 2600], 'z':[1500,1000,2000]})
ranges = pd.DataFrame({'ran': [5000, 5000, 5000, 5000, 5000]})


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("발사각도 예측")
        self.setWindowIcon(QIcon('icon.png'))
        
        #initialize data matrix
        self.wind_dir = pd.DataFrame({'wind_dir': list(range(6400)) ,
            'vec': [np.dot(sb.rotate_matrix( i * np.pi / 6400), np.array([0,1]))for i in range(6400)] })
        self.n_dic = {'cannon_x':n_cannons, 'cannon_y':n_cannons, 'cannon_z':n_cannons, 'balloon_x':n_balloons, 'balloon_y':n_balloons, 'balloon_z':n_balloons}
        self.data = {}
        self.data['cannon'] = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.data['balloon'] = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['balloon_x', 'balloon_y', 'balloon_z'])
        self.data['wind'] = pd.DataFrame(np.zeros((n_winds, 2)), index =winds_idx,  columns = ['wind_dir', 'wind_vel'])

 


        


  
        


        self.wind_dir_only = QIntValidator(0, 6399, self)
        self.int_only = QIntValidator(0,99999,self)
        
        
        #initialize plt plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
       
        
        #고사포와 풍선
       
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
        
        


        
        #고사총과 풍선 이름
        for i in range(n_cannons):
            exec('self.cannon_label_{} = QLabel("적 고사총 기지 {} ")'.format(i,i+1))
        for i in range(n_balloons):
            exec('self.balloon_label_{} = QLabel("아군 K-6 ({}) ")'.format(i,i+1))



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
            exec('self.cannon_alloc_{} = QLabel("아군 K-6 ({})")'.format(i, i+1))
            exec('self.alloc_{} = QLabel("-----")'.format(i))
            exec('self.balloon_alloc_{} = QLabel("적 고사총 기지 ")'.format(i))
            exec('self.idland_{} = QLabel("")'.format(i))
            exec('self.actland_{} = QLabel("")'.format(i))

        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        


######기본값 linedeit에 입력
        #고사포
        #표시
        self.cannon_x_0.setText("610"); self.cannon_y_0.setText("6400"); self.cannon_z_0.setText("150")
        self.cannon_x_1.setText("3350"); self.cannon_y_1.setText("8000"); self.cannon_z_1.setText("130")
        self.cannon_x_2.setText("4900"); self.cannon_y_2.setText("7700"); self.cannon_z_2.setText("220")
        self.cannon_x_3.setText("6700"); self.cannon_y_3.setText("8000"); self.cannon_z_3.setText("320")
        self.cannon_x_4.setText("8400"); self.cannon_y_4.setText("7200"); self.cannon_z_4.setText("220")

        #실제 데이터
        self.data['cannon'].iloc[0,:] = [610, 6400, 150]
        self.data['cannon'].iloc[1,:] = [3350, 8000, 130]
        self.data['cannon'].iloc[2,:] = [4900, 7700, 220]
        self.data['cannon'].iloc[3,:] = [6700, 8000, 320]
        self.data['cannon'].iloc[4,:] = [8400, 7200, 220]
        
        #아군 K-6
        #표시
        self.balloon_x_0.setText("2500"); self.balloon_y_0.setText("3900"); self.balloon_z_0.setText("150")
        self.balloon_x_1.setText("3200"); self.balloon_y_1.setText("5200"); self.balloon_z_1.setText("130")
        self.balloon_x_2.setText("5200"); self.balloon_y_2.setText("5300"); self.balloon_z_2.setText("220")
        self.balloon_x_3.setText("6500"); self.balloon_y_3.setText("5600"); self.balloon_z_3.setText("320")
        self.balloon_x_4.setText("7500"); self.balloon_y_4.setText("5300"); self.balloon_z_4.setText("220")

        #실제 데이터
        self.data['balloon'].iloc[0,:] = [2500, 3900, 150]
        self.data['balloon'].iloc[1,:] = [3200, 5200, 130]
        self.data['balloon'].iloc[2,:] = [5200, 5300, 220]
        self.data['balloon'].iloc[3,:] = [6500, 5600, 320]
        self.data['balloon'].iloc[4,:] = [7500, 5300, 220]

        #바람
            #표시
        self.wind_dir_0.setText("1000"); self.wind_vel_0.setText("10")
        self.wind_dir_1.setText("1500"); self.wind_vel_1.setText("4")
        self.wind_dir_2.setText("800"); self.wind_vel_2.setText("6")
        self.wind_dir_3.setText("6000"); self.wind_vel_3.setText("7")
        self.wind_dir_4.setText("4100"); self.wind_vel_4.setText("7")
        self.wind_dir_5.setText("3100"); self.wind_vel_5.setText("9")
        self.wind_dir_6.setText("4000"); self.wind_vel_6.setText("6")
        self.wind_dir_7.setText("370"); self.wind_vel_7.setText("7")
        self.wind_dir_8.setText("2150"); self.wind_vel_8.setText("11")
        self.wind_dir_9.setText("1980"); self.wind_vel_9.setText("8")
        self.wind_dir_10.setText("2870"); self.wind_vel_10.setText("4")
        self.wind_dir_11.setText("3010"); self.wind_vel_11.setText("6")
 
        #실제 데이터
        self.data['wind'].iloc[0,:] = [1000, 10]
        self.data['wind'].iloc[1,:] = [1500, 4]
        self.data['wind'].iloc[2,:] = [800, 6]
        self.data['wind'].iloc[3,:] = [6000, 7]
        self.data['wind'].iloc[4,:] = [4100, 7]
        self.data['wind'].iloc[5,:] = [3100, 9]
        self.data['wind'].iloc[6,:] = [4000, 6]
        self.data['wind'].iloc[7,:] = [370, 7]
        self.data['wind'].iloc[8,:] = [2150, 11]
        self.data['wind'].iloc[9,:] = [1980, 8]
        self.data['wind'].iloc[10,:] = [2870, 4]
        self.data['wind'].iloc[11,:] = [3010, 6]


 

        

        ##########################################################
        #left layout : 그림이 나오는 곳
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()

        # R_G1: 발사대
        R_G1 = QGroupBox('적 고사총 위치', self)  
        R_G1_box = QHBoxLayout()

        R_G1_box_G1 = QGroupBox('좌표', self)
        R_G1_box_G1_box = QHBoxLayout()

        R_G1_box_G1_box_0 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G1_box_0.addWidget(self.cannon_label_{})'.format(i))


        R_G1_box_G1_box_1 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G1_box_1.addWidget(self.cannon_x_{})'.format(i))
        R_G1_box_G1_box_2 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G1_box_G1_box_2.addWidget(self.cannon_y_{})'.format(i))
            
        R_G1_box_G1_box.addLayout(R_G1_box_G1_box_0)
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
        R_G2 = QGroupBox('아군 K-6 위치', self)  
        R_G2_box = QHBoxLayout()

        R_G2_box_G1 = QGroupBox('좌표', self)
        R_G2_box_G1_box = QHBoxLayout()

        R_G2_box_G1_box_0 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G1_box_0.addWidget(self.balloon_label_{})'.format(i))

        R_G2_box_G1_box_1 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G1_box_1.addWidget(self.balloon_x_{})'.format(i))
        R_G2_box_G1_box_2 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G2_box_G1_box_2.addWidget(self.balloon_y_{})'.format(i))
        R_G2_box_G1_box.addLayout(R_G2_box_G1_box_0)
        
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

        R_G4_box_G1 = QGroupBox('적 고사총 기지 - 풍선 매칭', self)
        R_G4_box_G1_box = QHBoxLayout()
        R_G4_box_G1_box_1 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G4_box_G1_box_1.addWidget(self.cannon_alloc_{})'.format(i))
        R_G4_box_G1_box_2 = QVBoxLayout()
        for i in range(n_balloons):
            exec('R_G4_box_G1_box_2.addWidget(self.alloc_{})'.format(i))
        R_G4_box_G1_box_3 = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G4_box_G1_box_3.addWidget(self.balloon_alloc_{})'.format(i))

        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_1)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_2)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_3)
        R_G4_box_G1.setLayout(R_G4_box_G1_box)

        R_G4_box_G2 = QGroupBox('발사각 계산', self)
        R_G4_box_G2_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('R_G4_box_G2_box.addWidget(self.idland_{})'.format(i))
        R_G4_box_G2.setLayout(R_G4_box_G2_box)



        R_G4_box.addWidget(R_G4_box_G1)
        R_G4_box.addWidget(R_G4_box_G2)
        
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
        #if sum(self.data['cannon'].cannon_z >= self.data['balloon'].balloon_z ) > 0:
            #QMessageBox.about(self, "오류", "포의 고도가 풍선의 고도보다 높거나 같으면 발사할 수 없습니다")
        #else:
        wind_tbl = pd.merge(self.data['wind'], self.wind_dir, how = 'left', on = 'wind_dir').set_index(self.data['wind'].index)
        self.ax.clear()
        per, idland, actland = pa.drawplot(100,  self.data['balloon'], self.data['cannon'],self.data['wind'], self.ax, ranges)
        print('idland:', idland)
        print('actland:', actland)


        for i, alloc in enumerate(per):
            exec('self.balloon_alloc_{}.setText("적 고사총 기지 {}")'.format(i, alloc + 1))
            exec('self.idland_{}.setText("{}")'.format(i, idland[i].round(2)))
            cannon = np.array(self.data['balloon'].iloc[i, :])
            enemy = np.array(self.data['cannon'].iloc[alloc, :])

            direc_init = (enemy[:2] - cannon[:2]) / la.norm(enemy[:2] - cannon[:2])
            #land = sb.optim(100, 45, direc_init, cannon, enemy, wind_tbl )

            under = la.norm(cannon[:2]- enemy[:2])


            theta_init = under * (9 / 1000)
            print('theta_init:', theta_init); print('direc_init:', direc_init)
                    
            def f(x):
                return sb.optim(100, x[0], np.array([x[1],x[2]]), cannon, enemy, wind_tbl)
            def constr1(x):
                x[1]
            def constr2(x):
                90 - x[1]
            def constr3(x):
                x[2]
            def constr4(x):
                90 - x[2]
            minimum = fmin_cobyla(f, [theta_init, direc_init[0], direc_init[1]], [constr1, constr2, constr3, constr4], rhoend=1e-7)
            print(minimum)
                # opt_eval = (sb.shoot_for_optim(100, cannon_0, sb.ang2coord(cannon[:1], minimum[0], minimum[1:], 5000), wind_tbl, 5000))
            exec('self.idland_{}.setText(str(round(minimum[0],2)))'.format(i))
  
        img = plt.imread("map.png")
        limit = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
        self.ax.imshow(img, extent=limit)   
        self.canvas.draw()

       
        # self.OptimOutput.setPlainText(str(minimum) + ' ' + str(opt_eval))
    
    
        
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


    def OptimAlgo(self):
        wind_tbl = pd.merge(self.data['wind'], self.wind_dir, how = 'left', on = 'wind_dir').set_index(self.data['wind'].index)
        cannon = np.array(self.data['balloon'].iloc[0, :])
        cannon_0 = np.append(cannon, 0)
        enemy = np.array(self.data['cannon'].iloc[0, :])

        direc_init = (enemy - cannon) / la.norm(enemy - cannon)
        print('cannon, enemy:', cannon, enemy)
        sb.optim(100, 45, direc_init, cannon, enemy, wind_tbl )

            # theta_init = np.arccos(under / ran )
            # print('theta_init:', theta_init)
            # 
            # print('direc_init:', direc_init)
                
            # def f(x):
            #     return (sb.optim(100, x[0], np.array([x[1],x[2]]), cannon_0, enemy, wind_tbl, ran))
            # def constr1(x):
            #     x[1]
            # def constr2(x):
            #     np.pi/2 - x[1]
            # def constr3(x):
            #     x[2]
            # def constr4(x):
            #     np.pi/2 - x[2]
            # minimum = fmin_cobyla(f, [theta_init, direc_init[0], direc_init[1]], [constr1, constr2, constr3, constr4], rhoend=1e-7)
            # opt_eval = (sb.shoot_for_optim(100, cannon_0, sb.ang2coord(cannon[:1], minimum[0], minimum[1:], 5000), wind_tbl, 5000))
            
if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()