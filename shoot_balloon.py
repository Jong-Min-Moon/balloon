import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations
import smallestenclosingcircle as sc
#data


col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan' , 'tab:purple', 'tab:brown', 'tab:pink']
wind_dir = {'wind_dir': [1,2,3,4,5,6,7,8],
            'vec':[ [0,1], [np.sqrt(0.5), np.sqrt(0.5)],
                    [1,0], [np.sqrt(0.5), -np.sqrt(0.5)],
                    [0,-1], [-np.sqrt(0.5), -np.sqrt(0.5)],
                    [-1,0], [-np.sqrt(0.5), np.sqrt(0.5)]]}
wind_dir = pd.DataFrame(wind_dir)

def optim(n_points, theta, v, cannon, enemy, wind_tbl, ran):
    virtual_balloon = ang2coord(cannon[:1], theta, v, ran)
    center = shoot_for_optim(n_points, cannon, virtual_balloon, wind_tbl, ran)
    return la.norm(center - enemy)

def rotate_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
def ang2coord(fire, theta, direc, ran):
    d = ran * np.cos(theta)
    balloon_z = ran * np.sin(theta)
    balloon_xyz = np.append((fire + d * direc), balloon_z)
    return balloon_xyz

def drawplot(n_iter, cannons, balloons, winds, ax, ranges):
    
    wind_tbl = pd.merge(winds, wind_dir, how = 'left', on = 'wind_dir').set_index(winds.index)
    #print(wind_tbl)

    idland = []; actland = []
    per, summ = allocate(cannons, balloons)
    for i, comb in enumerate(zip(range(len(cannons)), per)):
        #print('{}번째 발사. 포탄과 풍선 조합: {}'.format(i, comb))
        this_idland, this_actland = shoot( n_iter,  np.array(cannons.iloc[comb[0], :]), np.array(balloons.iloc[comb[1], :]), wind_tbl, ranges.iloc[i,0], ax, i)
        idland.append(this_idland); actland.append(this_actland)
    return per, idland, actland

def allocate(cannons, balloons):
    n = len(balloons)
    dist_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            a = np.array(cannons.iloc[i, :])
            b = np.array(balloons.iloc[j, :])
            dist_mat[i,j] = la.norm( a - b)   
    
    sum_min = np.inf
    permu_min = 0
    for permu in permutations(range(n), n):
        sum = 0
        for tup in (zip(range(n), permu)):
            sum += dist_mat[tup]
        if sum < sum_min:
            permu_min = permu
            sum_min = sum

    return permu_min, sum_min

def shoot(n_iter, cannon, balloon, wind_tbl, ran, ax, col_id):
    mycol = col_list[col_id]
    ax.scatter(cannon[0], cannon[1], color = mycol); ax.text(cannon[0], cannon[1], 'fire')
    ax.scatter(balloon[0], balloon[1], color = mycol); ax.text(balloon[0], balloon[1], 'balloon')

    x_values = []; y_values = [] #착탄점들의 중심점을 구하기 위해 다 저장
    #x_shootline = []; y_shootline = [] #착탄 경로 저장
    for i in range(n_iter):
        np.random.seed(i)
        #print('iteration {}'.format(i))
        idland, xy_now, peak_z, theta, direc_now = peak_xy(cannon, balloon, ran)
        #print('peak_z:', peak_z)
        if i == 0:
            ax.scatter(xy_now[0], xy_now[1], color = mycol); ax.text(xy_now[0], xy_now[1], 'peak') #최고점
            ax.scatter(idland[0], idland[1], s = 50, alpha = 0.7, color = mycol); ax.text(idland[0], idland[1], 'ideal landing') 
            ax.plot( [cannon[0], balloon[0]], [cannon[1], balloon[1]], color = mycol) #cannon to balloon
            ax.plot( [balloon[0], xy_now[0]], [balloon[1], xy_now[1]], color = mycol) #baloon to peak
            ax.plot( [xy_now[0], idland[0]], [xy_now[1], idland[1]], linestyle = '--', color = mycol) #peak to ideal landing
        

        # shoot
        #현재 고도보다 바로 위에 있는 테이블 값의 풍향을 사용할 것임
        wind_h = [idx[1] for idx in wind_tbl.index]; wind_h.insert(0,0)
        idx_now = (wind_h >= peak_z).tolist().index(True)
        h_now = peak_z

        #포탄 경로 그리기 준비      
        #total_move = 0
        while idx_now > 0:
            h_down = h_now - wind_h[idx_now - 1] #수직 하강 거리
            #print('고도 {} -> {}. {}만큼 하강하는 동안'.format(h_now, wind_h[idx_now - 1], h_down))

            one_step = h_down * np.tan(theta) #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
            #total_move += one_step
            #print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

            wind_vec = np.array(wind_tbl.vec[idx_now-1])
            wind_vel = wind_tbl.wind_vel[idx_now-1]
            interval_now = wind_tbl.index[idx_now - 1]
            #print('고도 {} in 구간 {}에서의 바람 방향: {}'.format(wind_h[idx_now], interval_now, wind_vec))
            #print('원래 전진 방향: {}'.format(direc_now))
            
            cossim = cos_sim(direc_now, wind_vec)
            vel_power = (100 + cossim * 20 * wind_vel) / 100
            one_step = one_step * vel_power #전진거리 수정
            #print('풍속에 의해 수정된 전진 거리: {}'.format(one_step))
            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = np.random.beta(2,30) * vel_power 
                direc_now = (1-w) * direc_now + w * wind_vec.astype('float64')
                direc_now =  (direc_now ) / la.norm(direc_now) #unit vector로 만들기
                #print('바람에 의해 수정된 방향: {}'.format(direc_now))  
            #else:
                #print('바람의 방향이 포탄의 방향과 정확히 일치(혹은 정확히 반대)하므로, 포탄 진행방향 변화 없음. 속도만 수정')


                
            #계산 종료.

            #포탄의 전진.
            xy_now = xy_now + direc_now * one_step
            #x_shootline.append(xy_now[0]); y_shootline.append(xy_now[1])  #peak에서 착탄까지 경로 저장
            #print('{},{}에 도착\n'.format(xy_now, wind_h[idx_now - 1]))
            
            idx_now -=1
            h_now = wind_h[idx_now]
            #print(xy_now)
        #print(x_shootline)
        
        x_values.append(xy_now[0])
        y_values.append(xy_now[1])

        
        ax.scatter(xy_now[0], xy_now[1], s = 10) #착탄지점 그래프에 그리기
        
    #print('실제로 총 {}만큼 전진'.format(total_move))
    #ax.plot(x_shootline, y_shootline) #꺾인 발사
    #ax.plot('xval', 'yval', data = pd.DataFrame({'xval': x_values, 'yval': y_values}), linestyle='none', markersize = 20) #scatterplot
    cir = sc.make_circle(zip(x_values, y_values))
    ax.text(cir[0], cir[1], 'center of actual landing') #최고점
    ax.add_patch( patches.Circle( (cir[0], cir[1]), # (x, y)
                                            cir[2], # radius
        alpha=0.4, 
        facecolor=mycol, 
        edgecolor=mycol, 
        linewidth=2, 
        linestyle='solid'))
  
    return idland, np.array([cir[0], cir[1]])


def shoot_for_optim(n_points, cannon, balloon, wind_tbl, ran):

    x_values = []; y_values = [] #착탄점들의 중심점을 구하기 위해 다 저장

    for i in range(n_points):
        np.random.seed(i)
        idland, xy_now, peak_z, theta, direc_now = peak_xy(cannon, balloon, ran)

        # shoot
        #현재 고도보다 바로 위에 있는 테이블 값의 풍향을 사용할 것임
        wind_h = [idx[1] for idx in wind_tbl.index]; wind_h.insert(0,0)
        idx_now = (wind_h >= peak_z).tolist().index(True)
        h_now = peak_z

        #포탄 경로 그리기 준비      
        while idx_now > 0:
            h_down = h_now - wind_h[idx_now - 1] #수직 하강 거리
            one_step = h_down * np.tan(theta) #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함

            wind_vec = np.array(wind_tbl.vec[idx_now-1])
            wind_vel = wind_tbl.wind_vel[idx_now-1]
            interval_now = wind_tbl.index[idx_now - 1]

            
            cossim = cos_sim(direc_now, wind_vec)
            vel_power = (100 + cossim * 20 * wind_vel) / 100
            one_step = one_step * vel_power #전진거리 수정

            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = np.random.beta(2,30) * vel_power 
                direc_now = (1-w) * direc_now + w * wind_vec.astype('float64')
                direc_now =  (direc_now ) / la.norm(direc_now) #unit vector로 만들기


                
            #계산 종료.

            #포탄의 전진.
            xy_now = xy_now + direc_now * one_step   
            idx_now -=1
            h_now = wind_h[idx_now]

        
        x_values.append(xy_now[0])
        y_values.append(xy_now[1])
      
    cir = sc.make_circle(zip(x_values, y_values))
    return np.array([cir[0], cir[1]])

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



def cos_sim(v1, v2):
    return (np.dot(v1,v2) / (la.norm(v1) * la.norm(v2))).round(7)

def vec2ang(v):
    return np.arctan2(v[1], v[0])




















