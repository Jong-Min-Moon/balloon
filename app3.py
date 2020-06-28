




ranges = pd.DataFrame({'ran': [4000, 4000, 4000]})

        
        #대포와 적
       
        self.enemy_x_label = QLabel('좌표')

  
        


        self.pushButton1 = QPushButton("발사각 계산하기(grid serch)")
        self.pushButton1.clicked.connect(self.GridSearch)


        

        






        ##########################################################
        #left layout : 그림이 나오는 곳
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.OptimOutput)
        leftLayout.addWidget(self.canvas)
        
        ##########################################################
        # right Layout : 입력창이 있는 곳
        rightLayout = QVBoxLayout()





        


        


     
       # 

 
        
        
    

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
            enemy_vec = np.array(self.data['enemy'].iloc[i,:])
            print(cannon_vec, enemy_vec)
            _,_,_, th, _ = sb.peak_xy(cannon_vec, enemy_vec, 5000)
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