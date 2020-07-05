import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QRegExp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import algorithm as alg
import pandas as pd
import numpy as np
from numpy import linalg as la
from scipy.optimize import fmin_cobyla

n_cannons = 5
n_K6s = 3
winds = np.linspace(0, 8000, num = 12 + 1) #고도 0미터부터 8000미터까지의 구간을 아래와 같이 12개 구간으로 나눔
winds_idx = [ (0,200), (201,500), (501,1000), (1001, 1500), (1501, 2000), (2001,2500), (2501,3000), (3001,4000), (4001,5000), (5001,6000), (6001,7000), (7001,8000) ]



class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("발사각도 예측")
        self.setWindowIcon(QIcon('icon.png'))
        
        #initialize data matrix
        self.wind_dir = pd.DataFrame({'wind_dir': [0] + list(range(1, 6400)[::-1]) ,
            'vec': [np.dot(alg.rotate_matrix( i * np.pi / 3200), np.array([0,1])) for i in range(6400)] })#남풍을 0으로 하여 바람 방향을 6400개로 나누고 각 방향의 unit vector를 저장

        self.int_input_widgets = {'cannon_z' : n_cannons, 'K6_z' : n_K6s}
        self.MGRS_input_widgets = {'cannon_x' : n_cannons, 'cannon_y':n_cannons, 'K6_x':n_K6s, 'K6_y':n_K6s}
        self.data = {}
        self.data['cannon'] = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.data['K6'] = pd.DataFrame(np.zeros((n_K6s, 3)), columns = ['K6_x', 'K6_y', 'K6_z'])
        self.data['wind'] = pd.DataFrame(np.zeros((12, 2)), index =winds_idx,  columns = ['wind_dir', 'wind_vel'])
 
        self.wind_dir_only = QIntValidator(0, 6399, self)
        self.int_only = QIntValidator(0,99999,self)
        self.MGRSvalid = QRegExpValidator(QRegExp("\d{5}"), self)
        
        #initialize plt plot
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.fig)
       
        
        #고사포와 풍선

        def setQlineEditLength(QLE, nchar): #QlineEdit의 너비를 조절하는 함수
            fm = QLE.fontMetrics()
            m = QLE.textMargins()
            c = QLE.contentsMargins()
            w = nchar * 1.3 * fm.width('x')+m.left()+m.right()+c.left()+c.right()
            QLE.setMaximumWidth(w + 8)

        self.LineEdit_list_cannon_K6 = {}
        for obj_name in self.int_input_widgets: #0-99999의 값을 input으로 받는 입력창 widget을 생성(적 고사포와 풍선의 높이)
            for i in range(self.int_input_widgets[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                exec('setQlineEditLength(self.{}_{}, 5)'.format(obj_name, i))
                exec('self.LineEdit_list_cannon_K6["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for obj_name in self.MGRS_input_widgets: #5자리 숫자를 input으로 받는 입력창 widget을 생성(적 고사포와 풍선의 MGRS 좌표)
            for i in range(self.MGRS_input_widgets[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.MGRSvalid)'.format(obj_name, i))
                exec('setQlineEditLength(self.{}_{}, 5)'.format(obj_name, i))
                exec('self.LineEdit_list_cannon_K6["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for widget in self.LineEdit_list_cannon_K6: #위에서 생성한 위젯에 입력 함수를 할당
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.LineEdit_list_cannon_K6[widget].textChanged.connect(lambda a = widget, this_widget = self.LineEdit_list_cannon_K6[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged(this_widget, this_i, this_j, this_mat ))
          
        


        
        #고사총과 풍선 이름
        for i in range(n_cannons):
            exec('self.cannon_label_{} = QLabel("적 고사총 기지 {} ")'.format(i,i+1))
        for i in range(n_K6s):
            exec('self.K6_label_{} = QLabel("아군 K-6 ({}) ")'.format(i,i+1))



        #바람
        
        self.height_label = QLabel('고도(m)')
        for i , idx in enumerate(self.data['wind'].index):
            exec('self.height_label_{} = QLabel("{}-{}")'.format(i, idx[0], idx[1]))
        
        self.wind_dir_list = {}
        self.wind_dir_label = QLabel('풍향(mil)')
        for i in range(12):
            exec('self.wind_dir_{} = QLineEdit()'.format(i))
            exec("self.wind_dir_list['self.wind_dir_{}'] = self.wind_dir_{}".format(i,i))
            exec('setQlineEditLength(self.wind_dir_{}, 5.4)'.format(i))
            exec('self.wind_dir_{}.setValidator(self.wind_dir_only)'.format(i))
        
        for widget in self.wind_dir_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            mat = self.data[inputs[0]]
            self.wind_dir_list[widget].textChanged.connect(lambda a = widget, this_widget = self.wind_dir_list[widget], this_i = i,  this_mat = mat: self.lineEditChanged2(this_widget, this_i, 0, this_mat ))
          

        self.wind_vel_list = {}
        self.wind_vel_label = QLabel('풍속')
        for i in range(12):
            exec('self.wind_vel_{} = QLineEdit()'.format(i))
            exec("self.wind_vel_list['self.wind_vel_{}'] = self.wind_vel_{}".format(i,i))
            exec('setQlineEditLength(self.wind_vel_{}, 4.4)'.format(i))
            exec('self.wind_vel_{}.setValidator(self.int_only)'.format(i))

        #대포 - 풍선 매칭
        for i in range(n_K6s):
            exec('self.cannon_alloc_{} = QLabel("아군 K-6 ({})")'.format(i, i+1))
            exec('self.alloc_{} = QLabel("-----")'.format(i))
            exec('self.K6_alloc_{} = QLabel("적 고사총 기지 ")'.format(i))
            exec('self.idland_{} = QLabel("")'.format(i))
            exec('self.actland_{} = QLabel("")'.format(i))

        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        


######기본값 linedeit에 입력
        #고사포
        #표시
        self.cannon_x_0.setText("00610"); self.cannon_y_0.setText("06400"); self.cannon_z_0.setText("150")
        self.cannon_x_1.setText("03350"); self.cannon_y_1.setText("08000"); self.cannon_z_1.setText("130")
        self.cannon_x_2.setText("04900"); self.cannon_y_2.setText("07700"); self.cannon_z_2.setText("220")
        self.cannon_x_3.setText("06700"); self.cannon_y_3.setText("08000"); self.cannon_z_3.setText("320")
        self.cannon_x_4.setText("08400"); self.cannon_y_4.setText("07200"); self.cannon_z_4.setText("220")

        #실제 데이터
        self.data['cannon'].iloc[0,:] = [610, 6400, 150]
        self.data['cannon'].iloc[1,:] = [3350, 8000, 130]
        self.data['cannon'].iloc[2,:] = [4900, 7700, 220]
        self.data['cannon'].iloc[3,:] = [6700, 8000, 320]
        self.data['cannon'].iloc[4,:] = [8400, 7200, 220]
        
        #아군 K-6
        #표시
        self.K6_x_0.setText("02500"); self.K6_y_0.setText("03900"); self.K6_z_0.setText("150")
        #self.K6_x_1.setText("3200"); self.K6_y_1.setText("5200"); self.K6_z_1.setText("130")
        self.K6_x_1.setText("05200"); self.K6_y_1.setText("05300"); self.K6_z_1.setText("220")
        self.K6_x_2.setText("06500"); self.K6_y_2.setText("05600"); self.K6_z_2.setText("320")
        #self.K6_x_4.setText("7500"); self.K6_y_4.setText("5300"); self.K6_z_4.setText("220")

        #실제 데이터
        self.data['K6'].iloc[0,:] = [2500, 3900, 150]
        #self.data['K6'].iloc[1,:] = [3200, 5200, 130]
        self.data['K6'].iloc[1,:] = [5200, 5300, 220]
        self.data['K6'].iloc[2,:] = [6500, 5600, 320]
        #self.data['K6'].iloc[4,:] = [7500, 5300, 220]

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


 
#########################################################################################################
####################################### LAYOUT ##########################################################
        

        ##########################################################
        #left layout : 그림이 나오는 곳

        # R_G4: 발사 결과
        R_G4 = QGroupBox('발사 결과', self)  
        R_G4_box = QHBoxLayout()

        R_G4_box_G1 = QGroupBox('아군 K6 - 적 고사총 기지 매칭', self)
        R_G4_box_G1_box = QHBoxLayout()
        R_G4_box_G1_box_1 = QVBoxLayout()
        for i in range(n_K6s):
            exec('R_G4_box_G1_box_1.addWidget(self.cannon_alloc_{})'.format(i))
        R_G4_box_G1_box_2 = QVBoxLayout()
        for i in range(n_K6s):
            exec('R_G4_box_G1_box_2.addWidget(self.alloc_{})'.format(i))
        R_G4_box_G1_box_3 = QVBoxLayout()
        for i in range(n_K6s):
            exec('R_G4_box_G1_box_3.addWidget(self.K6_alloc_{})'.format(i))

        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_1)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_2)
        R_G4_box_G1_box.addLayout(R_G4_box_G1_box_3)
        R_G4_box_G1.setLayout(R_G4_box_G1_box)

        R_G4_box_G2 = QGroupBox('사각(mil)', self)
        R_G4_box_G2_box = QVBoxLayout()
        for i in range(n_K6s):
            exec('R_G4_box_G2_box.addWidget(self.idland_{})'.format(i))
        R_G4_box_G2.setLayout(R_G4_box_G2_box)

        R_G4_box_G3 = QGroupBox('편각(mil)', self)
        R_G4_box_G3_box = QVBoxLayout()
        for i in range(n_K6s):
            exec('R_G4_box_G3_box.addWidget(self.actland_{})'.format(i))
        R_G4_box_G3.setLayout(R_G4_box_G3_box)


        R_G4_box.addWidget(R_G4_box_G1)
        R_G4_box.addWidget(R_G4_box_G2)
        R_G4_box.addWidget(R_G4_box_G3)
        R_G4.setLayout(R_G4_box)
        
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas, 5)
        leftLayout.addWidget(R_G4, 1)

        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()


        # R_G1: 적 고사총 위치
        R_G1 = QGroupBox('적 고사총 위치', self)  
        R_G1_box = QHBoxLayout()

        R_G1_box_G1 = QGroupBox('MGRS 좌표', self)
        R_G1_box_G1_box = QVBoxLayout()
        
        R_G1_box_G2 = QGroupBox('높이(m)', self)
        R_G1_box_G2_box = QVBoxLayout()

        for i in range(n_cannons):
            exec('R_G1_box_G1_box_{} = QHBoxLayout()'.format(i))
            exec('R_G1_box_G1_box_{}.addWidget(self.cannon_label_{})'.format(i, i)) #'적 고사총 기지'
            exec('R_G1_box_G1_box_{}.addWidget(self.cannon_x_{})'.format(i, i)) # x좌표
            exec('R_G1_box_G1_box_{}.addWidget(self.cannon_y_{})'.format(i, i)) # y좌표
            exec('R_G1_box_G1_box.addLayout(R_G1_box_G1_box_{})'.format(i))
            exec('R_G1_box_G2_box.addWidget(self.cannon_z_{})'.format(i)) #높이
        R_G1_box_G1.setLayout(R_G1_box_G1_box)
        R_G1_box_G2.setLayout(R_G1_box_G2_box)     

        R_G1_box.addWidget(R_G1_box_G1, 4)
        R_G1_box.addWidget(R_G1_box_G2, 1)
       
        R_G1.setLayout(R_G1_box)


        # R_G2: 아군 K-6
        R_G2 = QGroupBox('아군 K-6 위치', self)
        R_G2_box = QHBoxLayout()

        R_G2_box_G1 = QGroupBox('MGRS 좌표', self)
        R_G2_box_G1_box = QVBoxLayout()

        R_G2_box_G2 = QGroupBox('높이(m)', self)
        R_G2_box_G2_box = QVBoxLayout()

        for i in range(n_K6s):
            exec('R_G2_box_G1_box_{} = QHBoxLayout()'.format(i))
            exec('R_G2_box_G1_box_{}.addWidget(self.K6_label_{})'.format(i, i)) #'적 고사총 기지'
            exec('R_G2_box_G1_box_{}.addWidget(self.K6_x_{})'.format(i, i)) # x좌표
            exec('R_G2_box_G1_box_{}.addWidget(self.K6_y_{})'.format(i, i)) # y좌표
            exec('R_G2_box_G1_box.addLayout(R_G2_box_G1_box_{})'.format(i))      
            exec('R_G2_box_G2_box.addWidget(self.K6_z_{})'.format(i,i))
        
        R_G2_box_G1.setLayout(R_G2_box_G1_box)
        R_G2_box_G2.setLayout(R_G2_box_G2_box)

        R_G2_box.addWidget(R_G2_box_G1, 4)
        R_G2_box.addWidget(R_G2_box_G2, 1)

        R_G2.setLayout(R_G2_box)
        

        #R3 : 바람
        R3 = QGroupBox('고도별 풍향과 풍속', self)
        R3_box = QHBoxLayout()

        R3_box_G1 = QGroupBox('고도(m)', self)
        R3_box_G1_box = QVBoxLayout()

        R3_box_G2 = QGroupBox('풍향(mil)', self)
        R3_box_G2_box = QVBoxLayout()

        R3_box_G3 = QGroupBox('풍속(m/s)', self)
        R3_box_G3_box = QVBoxLayout()
        for i in range(12):
            exec('R3_box_G1_box.addWidget(self.height_label_{})'.format(i))#고도
            exec('R3_box_G2_box.addWidget(self.wind_dir_{})'.format(i))#풍향 
            exec('R3_box_G3_box.addWidget(self.wind_vel_{})'.format(i))#풍속
        R3_box_G1.setLayout(R3_box_G1_box)
        R3_box_G2.setLayout(R3_box_G2_box)
        R3_box_G3.setLayout(R3_box_G3_box)

        R3_box.addWidget(R3_box_G1, 3)
        R3_box.addWidget(R3_box_G2, 2)
        R3_box.addWidget(R3_box_G3, 2)

        R3.setLayout(R3_box)








        

        rightLayout.addWidget(R_G1)
        rightLayout.addStretch(2)
        rightLayout.addWidget(R_G2)
        rightLayout.addStretch(2)
        rightLayout.addWidget(R3)
        rightLayout.addStretch(2)
     
        rightLayout.addWidget(self.pushButton)
     


 
        
        
     
     


        layout = QHBoxLayout()
        layout.addLayout(leftLayout, 3)
        layout.addLayout(rightLayout, 1)
        
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def pushButtonClicked(self):
        #if sum(self.data['cannon'].cannon_z >= self.data['K6'].K6_z ) > 0:
            #QMessageBox.about(self, "오류", "포의 고도가 풍선의 고도보다 높거나 같으면 발사할 수 없습니다")
        #else:

        #그래프 축 제거 및 여백 정리
        self.ax.clear()
        self.ax.axis('off')
        plt.xticks([]); plt.yticks([])
        self.fig.tight_layout()

        wind_tbl = pd.merge(self.data['wind'], self.wind_dir, how = 'left', on = 'wind_dir').set_index(self.data['wind'].index)

        per = alg.allocate(self.data['K6'], self.data['cannon'])
        alg.draw_ideal(per, self.data['K6'], self.data['cannon'], self.ax)



        for i, alloc in enumerate(per):
            exec('self.K6_alloc_{}.setText("적 고사총 기지 {}")'.format(i, alloc + 1))
            cannon = np.array(self.data['K6'].iloc[i, :])
            enemy = np.array(self.data['cannon'].iloc[alloc, :])


        


            direc_init = (enemy[:2] - cannon[:2]) / la.norm(enemy[:2] - cannon[:2])

            under = la.norm(cannon[:2]- enemy[:2])


            theta_init = under * (9 / 1000)
                    
            def f(x):
                return alg.optim(100, x[0], np.array([x[1],x[2]]), cannon, enemy, wind_tbl)
            def constr1(x):
                x[1]
            def constr2(x):
                90 - x[1]
            def constr3(x):
                x[2]
            def constr4(x):
                90 - x[2]
            #minimum = fmin_cobyla(f, [theta_init, direc_init[0], direc_init[1]], [constr1, constr2, constr3, constr4], rhoend=1e-7)
            #exec('self.idland_{}.setText(str(    int(round(    minimum[0] * 6400 / 360  , 0))  ))'.format(i))
            #exec('self.actland_{}.setText(str( alg.vec2mil(minimum[1:]) ))'.format(i))


        img = plt.imread("map.png")
        limit = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
        self.ax.imshow(img, extent=limit)   
        
        
        #사거리 내의 적 고사포에 점선을 연결해 주기
        available = alg.allocate2(self.data['K6'], self.data['cannon'])
        for i, alist in enumerate(available):
            print(alist)
            for aenemy in alist:
                x = self.data['K6'].iloc[i,:]
                y = self.data['cannon'].iloc[aenemy,:]
                self.ax.plot( [x[0], y[0]], [x[1], y[1]], linestyle = '--', color = 'royalblue') #peak to ideal landing
        self.canvas.draw()
    
        
    def lineEditChanged(self, widget, i, j, mat):
        try:
            mat.loc[i,j] = int(widget.text())
            print(mat)
        except:
            print('입력이 없음')
      
    def lineEditChanged2(self, widget, i, j, mat):
        try:
            mat.iloc[i,j] = int(widget.text())
        except:
            print('입력이 없음')


            
if __name__ == "__main__":


    app = QApplication(sys.argv)

    id = QFontDatabase.addApplicationFont("NanumBarunGothicBold.ttf")
    _fontstr = QFontDatabase.applicationFontFamilies(id)[0]
    font = QFont(_fontstr, 12)
    app.setFont(font)

    window = MyWindow()
    window.show()
    app.exec_()