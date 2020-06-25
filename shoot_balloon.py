import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt


#data

wind_dic = {'h': [0,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000],
            'dir':[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
            'vel':[1,2,1,3,4,5,1,7,5,3,9,10,4,5,6,2]
}
wind_tbl = pd.DataFrame(wind_dic)

wind_dir = {'dir': [1,2,3,4,5,6,7,8],
            'vec':[ [0,1], [np.sqrt(0.5), np.sqrt(0.5)],
                    [1,0], [np.sqrt(0.5), -np.sqrt(0.5)],
                    [0,-1], [-np.sqrt(0.5), -np.sqrt(0.5)],
                    [-1,0], [-np.sqrt(0.5), np.sqrt(0.5)]]}
wind_dir = pd.DataFrame(wind_dir)

wind_tbl = pd.merge(wind_tbl, wind_dir, how = 'left', on = 'dir')

#functions
def cos_sim(v1, v2):
    return round(np.dot(v1,v2) / (la.norm(v1) * la.norm(v2)),7)

def vec2ang(v):
    return np.arctan2(v[1], v[0])

def peak_xy(start_xyz, target_xyz, ran):
    start_xy = start_xyz[:2]
    target_xy = target_xyz[:2]
    h = target_xyz[2] - start_xyz[2]

    d1 = la.norm(start_xy - target_xy)
    s = np.sqrt(d1**2 + h**2)
    d2 = ran * (d1/s)
    peak_z = ran * (h/s)

    xy_shftd = target_xy - start_xy
    cos_and_sin = xy_shftd / d1
    peak_xy_shftd = d2 * cos_and_sin
    peak_xy = peak_xy_shftd + start_xy

    theta = np.arccos(h/s)
    direc = xy_shftd / la.norm(xy_shftd)

    ideal_landing = peak_xy + peak_xy_shftd
    #print('바람이 없을 시 예상 착륙점: {}'.format(ideal_landing))
    #print('총 이동 거리: {}에서 0까지 내려가는 동안 {}\n'.format(peak_z, peak_z * np.tan(theta)))
    return ideal_landing, peak_xy, peak_z, theta, direc






def shoot(n_iter, cannon, balloon, ran):
    plt.scatter(cannon[0], cannon[1])
    plt.text(cannon[0], cannon[1], 'fire')
    plt.scatter(balloon[0], balloon[1])
    plt.text(balloon[0], balloon[1], 'balloon')

    for i in range(n_iter):
        idland, xy_now, peak_z, theta, direc_now = peak_xy(cannon, balloon, ran)

        if i == 0:
            print('peak', xy_now)
            plt.scatter(xy_now[0], xy_now[1])
            plt.text(xy_now[0], xy_now[1], 'peak')

            plt.plot( [cannon[0], balloon[0]], [cannon[1], balloon[1]]) #cannon to balloon
            plt.plot( [balloon[0], xy_now[0]], [balloon[1], xy_now[1]])
            plt.plot( [xy_now[0], idland[0]], [xy_now[1], idland[1]]) #peak to ideal landing
        

    # shoot
        #현재 고도보다 바로 위에 있는 테이블 값의 풍향을 사용할 것임
        idx_now = (wind_tbl.h >= peak_z).tolist().index(True)
        h_now = peak_z

        #포탄 경로 그리기 준비
        x_values = [xy_now[0]]
        y_values = [xy_now[1]]

        total_move = 0
        while idx_now > 0:
            h_down = h_now - wind_tbl.h[idx_now - 1] #수직 하강 거리
            #print('고도 {} -> {}. {}만큼 하강하는 동안'.format(h_now, wind_tbl.h[idx_now - 1], h_down))

            one_step = h_down * np.tan(theta) #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
            total_move += one_step
            #print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

            wind_vec = np.array(wind_tbl.vec[idx_now])
            wind_vel = wind_tbl.vel[idx_now]
            #print('고도 {}에서의 바람 방향: {}'.format(wind_tbl.h[idx_now], wind_vec))
            #print('원래 전진 방향: {}'.format(direc_now))
            
            cossim = cos_sim(direc_now, wind_vec)
            vel_power = (100 + cossim * 6 * wind_vel) / 100
            one_step = one_step * vel_power #전진거리 수정
            #print('풍속에 의해 수정된 전진 거리: {}'.format(one_step))
            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = np.random.beta(2,30) * vel_power 
                direc_now = (1-w) * direc_now + w * wind_vec.astype('float64')
                direc_now =  (direc_now ) / la.norm(direc_now) #unit vector로 만들기
                
            #else:
                #print('바람의 방향이 포탄의 방향과 정확히 일치(혹은 정확히 반대)하므로, 포탄 진행방향 변화 없음. 속도만 수정')


                #print('바람에 의해 수정된 방향: {}'.format(direc_now))
            #계산 종료.

            #포탄의 전진.
            xy_now = xy_now + direc_now * one_step
            #print('{},{}에 도착\n'.format(xy_now, wind_tbl.h[idx_now - 1]))
            
            idx_now -=1
            h_now = wind_tbl.h[idx_now]
            
            x_values.append(xy_now[0])
            y_values.append(xy_now[1])

        plt.scatter(xy_now[0], xy_now[1])
        plt.plot(x_values, y_values) #꺾인 발사
    #print('실제로 총 {}만큼 전진'.format(total_move))

    plt.scatter(idland[0], idland[1], s = 500) #이상적인 발사
    

shoot( 100,  np.array([0,0,1000]), np.array([np.sqrt(2) * 4000, 4000, 4000]), 8000)
shoot(100, np.array([-1000,1000,500]), np.array([-3000,-3000,1200]), 5000)
plt.show()