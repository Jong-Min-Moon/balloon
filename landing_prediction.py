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
import qtmodern.styles
import qtmodern.windows

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
        
        #initialize data matrix
        self.int_input_widgets = {'cannon_z' : n_cannons, 'balloon_z' : n_balloons}
        self.MGRS_input_widgets = {'cannon_x' : n_cannons, 'cannon_y':n_cannons, 'balloon_x':n_balloons, 'balloon_y':n_balloons}
        self.data = {}
        self.data['cannon'] = pd.DataFrame(np.zeros((n_cannons, 3)), columns = ['cannon_x', 'cannon_y', 'cannon_z'])
        self.data['balloon'] = pd.DataFrame(np.zeros((n_balloons, 3)), columns = ['balloon_x', 'balloon_y', 'balloon_z'])
        self.data['wind'] = pd.DataFrame(np.zeros((12, 2)), index =winds_idx,  columns = ['wind_dir', 'wind_vel'])

 




        #입력값 validators
        self.wind_dir_only = QIntValidator(0, 6399, self)
        self.int_only = QIntValidator(0,99999,self)
        self.MGRSvalid = QRegExpValidator(QRegExp("\d{5}"), self)
        
        #initialize plt plot
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.fig)
       
        
        #고사포와 풍선 좌표 및 높이 입력 위젯 생성
        
        def setQlineEditLength(QLE, nchar): #QlineEdit의 너비를 조절하는 함수
            fm = QLE.fontMetrics()
            m = QLE.textMargins()
            c = QLE.contentsMargins()
            w = nchar * 1.3 * fm.width('x')+m.left()+m.right()+c.left()+c.right()
            QLE.setMaximumWidth(w + 8)

        self.LineEdit_list_cannon_balloon = {}
        for obj_name in self.int_input_widgets: #0-99999의 값을 input으로 받는 입력창 widget을 생성(적 고사포와 풍선의 높이)
            for i in range(self.int_input_widgets[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.int_only)'.format(obj_name, i))
                exec('setQlineEditLength(self.{}_{}, 5)'.format(obj_name, i))
                exec('self.LineEdit_list_cannon_balloon["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for obj_name in self.MGRS_input_widgets: #5자리 숫자를 input으로 받는 입력창 widget을 생성(적 고사포와 풍선의 MGRS 좌표)
            for i in range(self.MGRS_input_widgets[obj_name]):
                exec('self.{}_{} = QLineEdit()'.format(obj_name, i))
                exec('self.{}_{}.setValidator(self.MGRSvalid)'.format(obj_name, i))
                exec('setQlineEditLength(self.{}_{}, 5)'.format(obj_name, i))
                exec('self.LineEdit_list_cannon_balloon["self.{}_{}"] = self.{}_{}'.format(obj_name, i, obj_name, i))
        for widget in self.LineEdit_list_cannon_balloon: #위에서 생성한 위젯에 입력 함수를 할당
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            j = inputs[0] + '_' + inputs[1]
            mat = self.data[inputs[0]]
            self.LineEdit_list_cannon_balloon[widget].textChanged.connect(lambda a = widget, this_widget = self.LineEdit_list_cannon_balloon[widget], this_i = i, this_j = j, this_mat = mat: self.lineEditChanged(this_widget, this_i, this_j, this_mat ))
        
        


        
        #고사총과 풍선 이름
        for i in range(n_cannons):
            exec('self.cannon_label_{} = QLabel("적 고사총 기지 {} ")'.format(i,i+1))
        for i in range(n_balloons):
            exec('self.balloon_label_{} = QLabel("풍선 {} ")'.format(i,i+1))



        #바람
        
        self.height_label = QLabel('고도(m)')
        for i , idx in enumerate(self.data['wind'].index):
            exec('self.height_label_{} = QLabel("{}-{}")'.format(i, idx[0], idx[1]))
        
        self.wind_dir_list = {}
        self.wind_dir_label = QLabel('풍향(mil)')
        for i in range(12):
            exec('self.wind_dir_{} = QLineEdit()'.format(i))
            exec("self.wind_dir_list['self.wind_dir_{}'] = self.wind_dir_{}".format(i,i))
            exec('setQlineEditLength(self.wind_dir_{}, 4)'.format(i))
     
        
        for widget in self.wind_dir_list:
            inputs = widget.replace('self.', '').split('_')
            i = int(inputs[2])
            mat = self.data[inputs[0]]
            self.wind_dir_list[widget].textChanged.connect(lambda a = widget, this_widget = self.wind_dir_list[widget], this_i = i,  this_mat = mat: self.lineEditChanged2(this_widget, this_i, 0, this_mat ))
          

        self.wind_vel_list = {}
        self.wind_vel_label = QLabel('풍속(m/s)')
        for i in range(12):
            exec('self.wind_vel_{} = QLineEdit()'.format(i))
            exec("self.wind_vel_list['self.wind_vel_{}'] = self.wind_vel_{}".format(i,i))
            exec('setQlineEditLength(self.wind_vel_{}, 3)'.format(i))

        #대포 - 풍선 매칭
        for i in range(n_cannons):
            exec('self.cannon_alloc_{} = QLabel("적 고사총 기지 {}")'.format(i, i+1))
            exec('self.alloc_{} = QLabel("-----")'.format(i))
            exec('self.balloon_alloc_{} = QLabel("풍선 ")'.format(i))
            exec('self.idland_{} = QLabel("")'.format(i))
            exec('self.actland_{} = QLabel("")'.format(i))

        self.pushButton = QPushButton("차트그리기")
        self.pushButton.clicked.connect(self.pushButtonClicked)

        


######기본값 linedeit에 입력
        #고사포
        #표시
        exec('self.cannon_x_0.setText("00610"); self.cannon_y_0.setText("06400"); self.cannon_z_0.setText("150")')
        self.cannon_x_1.setText("03350"); self.cannon_y_1.setText("08000"); self.cannon_z_1.setText("200")
        self.cannon_x_2.setText("04900"); self.cannon_y_2.setText("07700"); self.cannon_z_2.setText("60")
        self.cannon_x_3.setText("06700"); self.cannon_y_3.setText("08600"); self.cannon_z_3.setText("500")
        self.cannon_x_4.setText("08400"); self.cannon_y_4.setText("07200"); self.cannon_z_4.setText("320")

        #실제 데이터
        self.data['cannon'].iloc[0,:] = [610, 6400, 150]
        self.data['cannon'].iloc[1,:] = [3350, 8000, 200]
        self.data['cannon'].iloc[2,:] = [4900, 7700, 60]
        self.data['cannon'].iloc[3,:] = [6700, 8600, 500]
        self.data['cannon'].iloc[4,:] = [8400, 7200, 320]
        
        #풍선
        #표시
        self.balloon_x_0.setText("02500"); self.balloon_y_0.setText("03900"); self.balloon_z_0.setText("4000")
        self.balloon_x_1.setText("04500"); self.balloon_y_1.setText("05200"); self.balloon_z_1.setText("4000")
        self.balloon_x_2.setText("07500"); self.balloon_y_2.setText("05300"); self.balloon_z_2.setText("4200")
        #실제 데이터
        self.data['balloon'].iloc[0,:] = [2500, 3900, 4000]
        self.data['balloon'].iloc[1,:] = [4500, 5200, 4000]
        self.data['balloon'].iloc[2,:] = [7500, 5300, 4200]


        #바람
            #표시
        self.wind_dir_0.setText("3200"); self.wind_vel_0.setText("10")
        self.wind_dir_1.setText("4200"); self.wind_vel_1.setText("4")
        self.wind_dir_2.setText("3400"); self.wind_vel_2.setText("6")
        self.wind_dir_3.setText("4000"); self.wind_vel_3.setText("7")
        self.wind_dir_4.setText("4100"); self.wind_vel_4.setText("7")
        self.wind_dir_5.setText("3100"); self.wind_vel_5.setText("9")
        self.wind_dir_6.setText("4000"); self.wind_vel_6.setText("6")
        self.wind_dir_7.setText("4300"); self.wind_vel_7.setText("7")
        self.wind_dir_8.setText("4150"); self.wind_vel_8.setText("11")
        self.wind_dir_9.setText("3800"); self.wind_vel_9.setText("8")
        self.wind_dir_10.setText("3870"); self.wind_vel_10.setText("4")
        self.wind_dir_11.setText("4010"); self.wind_vel_11.setText("6")
 
        #실제 데이터
        self.data['wind'].iloc[0,:] = [3200, 10]
        self.data['wind'].iloc[1,:] = [4200, 4]
        self.data['wind'].iloc[2,:] = [3400, 6]
        self.data['wind'].iloc[3,:] = [4000, 7]
        self.data['wind'].iloc[4,:] = [4100, 7]
        self.data['wind'].iloc[5,:] = [3100, 9]
        self.data['wind'].iloc[6,:] = [4000, 6]
        self.data['wind'].iloc[7,:] = [4300, 7]
        self.data['wind'].iloc[8,:] = [4150, 11]
        self.data['wind'].iloc[9,:] = [3800, 8]
        self.data['wind'].iloc[10,:] = [3870, 4]
        self.data['wind'].iloc[11,:] = [4010, 6]



 

        

#########################################################################################################
####################################### LAYOUT ##########################################################
        #left layout : 그림이 나오는 곳

                # R_G4: 발사 결과
        L_G2 = QGroupBox('발사 결과', self)  
        L_G2_box = QHBoxLayout()

        L_G2_box_G1 = QGroupBox('적 고사총 기지 - 풍선 매칭', self)
        L_G2_box_G1_box = QHBoxLayout()
        L_G2_box_G1_box_1 = QVBoxLayout()
        for i in range(n_cannons):
            exec('L_G2_box_G1_box_1.addWidget(self.cannon_alloc_{})'.format(i))
        L_G2_box_G1_box_2 = QVBoxLayout()
        for i in range(n_cannons):
            exec('L_G2_box_G1_box_2.addWidget(self.alloc_{})'.format(i))
        L_G2_box_G1_box_3 = QVBoxLayout()
        for i in range(n_cannons):
            exec('L_G2_box_G1_box_3.addWidget(self.balloon_alloc_{})'.format(i))

        L_G2_box_G1_box.addLayout(L_G2_box_G1_box_1)
        L_G2_box_G1_box.addLayout(L_G2_box_G1_box_2)
        L_G2_box_G1_box.addLayout(L_G2_box_G1_box_3)
        L_G2_box_G1.setLayout(L_G2_box_G1_box)

        L_G2_box_G2 = QGroupBox('이론적 낙탄 지점(MGRS 좌표)', self)
        L_G2_box_G2_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('L_G2_box_G2_box.addWidget(self.idland_{})'.format(i))
        L_G2_box_G2.setLayout(L_G2_box_G2_box)

        L_G2_box_G3 = QGroupBox('예측된 낙탄 지점(MGRS 좌표)', self)
        L_G2_box_G3_box = QVBoxLayout()
        for i in range(n_cannons):
            exec('L_G2_box_G3_box.addWidget(self.actland_{})'.format(i))
        L_G2_box_G3.setLayout(L_G2_box_G3_box)

        L_G2_box.addWidget(L_G2_box_G1)
        L_G2_box.addWidget(L_G2_box_G2)
        L_G2_box.addWidget(L_G2_box_G3)
        L_G2.setLayout(L_G2_box)
       
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas,5)
        leftLayout.addWidget(L_G2,1)
        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()

        # R_G1: 발사대
        R_G1 = QGroupBox('적 고사총 위치', self)  
        R_G1_box = QHBoxLayout()

        R_G1_box_G1 = QGroupBox('MGRS 좌표', self)
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

        R_G1_box_G2 = QGroupBox('높이(m)', self)
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

        R_G2_box_G1 = QGroupBox('MGRS 좌표', self)
        R_G2_box_G1_box = QVBoxLayout()

        for i in range(n_balloons):
            exec('R_G2_box_G1_box_{} = QHBoxLayout()'.format(i))
            exec('R_G2_box_G1_box_{}.addWidget(self.balloon_label_{})'.format(i, i)) #'적 고사총 기지'
            exec('R_G2_box_G1_box_{}.addWidget(self.balloon_x_{})'.format(i, i)) # x좌표
            exec('R_G2_box_G1_box_{}.addWidget(self.balloon_y_{})'.format(i, i)) # y좌표
            exec('R_G2_box_G1_box.addLayout(R_G2_box_G1_box_{})'.format(i))      
        R_G2_box_G1.setLayout(R_G2_box_G1_box)





        R_G2_box_G2 = QGroupBox('높이(m)', self)
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
        for i in range(12):
            exec('R3box_1.addWidget(self.height_label_{})'.format(i))
    
        R3box_2 = QVBoxLayout()
        R3box_2.addWidget(self.wind_dir_label)
        for i in range(12):
            exec('R3box_2.addWidget(self.wind_dir_{})'.format(i))
            exec('self.wind_dir_{}.setValidator(self.wind_dir_only)'.format(i))

        
        R3box_3 = QVBoxLayout()
        R3box_3.addWidget(self.wind_vel_label)
        for i in range(12):
            exec('R3box_3.addWidget(self.wind_vel_{})'.format(i))
            exec('self.wind_vel_{}.setValidator(self.int_only)'.format(i))
        R3box.addLayout(R3box_1)
        R3box.addLayout(R3box_2)
        R3box.addLayout(R3box_3)
        
        R3.setLayout(R3box)


        

        rightLayout.addWidget(R_G1)
        #rightLayout.addStretch(3)
        rightLayout.addWidget(R_G2)
        #rightLayout.addStretch(3)
        rightLayout.addWidget(R3)
        #rightLayout.addStretch(3)
        
        rightLayout.addWidget(self.pushButton)
     
       # 

 
        
        
     
     


        layout = QHBoxLayout()
        layout.addLayout(leftLayout, 3)
        layout.addLayout(rightLayout,1)
        
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def pushButtonClicked(self):
        #if sum(self.data['cannon'].cannon_z >= self.data['balloon'].balloon_z ) > 0:
            #QMessageBox.about(self, "오류", "포의 고도가 풍선의 고도보다 높거나 같으면 발사할 수 없습니다")
        #else:
        self.ax.clear(); self.ax.axis('off')
        per, idland, actland = sb.drawplot(50, self.data['cannon'], self.data['balloon'], self.data['wind'], self.ax)
        #for i in range(len(per)):
        for i, alloc in enumerate(per):

            exec('self.balloon_alloc_{}.setText("풍선 {}")'.format(i, alloc + 1))
            exec('self.idland_{}.setText("{}")'.format(i, sb.five_digits(idland[i])))
            exec('self.actland_{}.setText("{}")'.format(i, sb.five_digits(actland[i])))
        img = plt.imread("map.png")
        limit = list(self.ax.get_xlim()) + list(self.ax.get_ylim())
        self.ax.imshow(img, extent=limit)   
        self.canvas.draw()




    def lineEditChanged(self, widget, i, j, mat): #lineEdit위젯에 입력한 고사포 및 풍선 값이 바뀔 경우 내부의 데이터프레임에서 즉시 반영하는 함수. 
        try:
            mat.loc[i,j] = int(widget.text())
            print(mat)
        except:
            print('입력이 없음')
      
    def lineEditChanged2(self, widget, i, j, mat): #lineEdit위젯에 입력한 바람 값이 바뀔 경우 내부의 데이터프레임에서 즉시 반영하는 함수. 
        try:
            mat.iloc[i,j] = int(widget.text())
        except:
            print('입력이 없음')


if __name__ == "__main__":

    
    app = QApplication(sys.argv)
    
    id = QFontDatabase.addApplicationFont("NanumBarunGothic.ttf")
    _fontstr = QFontDatabase.applicationFontFamilies(id)[0]
    font = QFont(_fontstr, 12)
    app.setFont(font)

    window = MyWindow()
    #qtmodern.styles.light(app)
    window.show()
    app.exec_()