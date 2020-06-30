import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations
import smallestenclosingcircle as sc
#data
def rotate_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

col_list = ['indianred', 'firebrick', 'maroon', 'red', 'crimson' , 'orangered', 'tab:brown', 'tab:pink']
wind_dir = {'wind_dir': list(range(6400)[::-1]) ,
            'vec': [np.dot(rotate_matrix( i * np.pi / 6400), np.array([0,1]))for i in range(6400)] }
            # 'vec':[ [0,1], [np.sqrt(0.5), np.sqrt(0.5)],
            #         [1,0], [np.sqrt(0.5), -np.sqrt(0.5)],
            #         [0,-1], [-np.sqrt(0.5), -np.sqrt(0.5)],
            #         [-1,0], [-np.sqrt(0.5), np.sqrt(0.5)]]}
wind_dir = pd.DataFrame(wind_dir)



def optim(n_points, theta, v, cannon, enemy, wind_tbl, ran):
    virtual_balloon = ang2coord(cannon[:1], theta, v, ran)
    center = shoot_for_optim(n_points, cannon, virtual_balloon, wind_tbl, ran)
    return la.norm(center - enemy)

    
def ang2coord(fire, theta, direc, ran):
    d = ran * np.cos(theta)
    balloon_z = ran * np.sin(theta)
    balloon_xyz = np.append((fire + d * direc), balloon_z)
    return balloon_xyz

def drawplot(n_iter, cannons, balloons, winds, ax, ranges):
    
    wind_tbl = pd.merge(winds, wind_dir, how = 'left', on = 'wind_dir').set_index(winds.index)
    print(wind_tbl)

    idland = []; actland = []
    per = allocate(cannons, balloons)
    print(list(zip(range(len(cannons)), per)))
    for i, comb in enumerate(zip(range(len(cannons)), per)):
        print('{}번째 발사. 포탄과 풍선 조합: {}'.format(i, comb))
        print('1', cannons.iloc[comb[0], :])
        print('2', balloons.iloc[comb[1], :])
        print('3',  ranges.iloc[i,0])
        this_idland, this_actland = shoot( n_iter,  np.array(cannons.iloc[comb[0], :]), np.array(balloons.iloc[comb[1], :]), wind_tbl,  ax, i)
        idland.append(this_idland); actland.append(this_actland)
    return per, idland, actland

def allocate(cannons, balloons):
    m = len(cannons)
    n = len(balloons)
    dist_mat = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            a = np.array(cannons.iloc[i, :])
            b = np.array(balloons.iloc[j, :])
            dist_mat[i,j] = la.norm( a - b)   
    match = []
    for i in range(m):
        distances = dist_mat[i,:]
        match.append(distances.argmin())


    # sum_min = np.inf
    # permu_min = 0
    # for permu in permutations(range(n), n):
    #     sum = 0
    #     for tup in (zip(range(n), permu)):
    #         sum += dist_mat[tup]
    #     if sum < sum_min:
    #         permu_min = permu
    #         sum_min = sum

    return match

def shoot(n_iter, cannon, balloon, wind_tbl, ax, col_id):
    mycol = col_list[col_id]
    ax.scatter(cannon[0], cannon[1], color = mycol); ax.text(cannon[0], cannon[1], 'K-6')
    ax.scatter(balloon[0], balloon[1], color = mycol, s = 300); ax.text(balloon[0], balloon[1], 'enemy')

    x_values = []; y_values = [] #착탄점들의 중심점을 구하기 위해 다 저장
    #x_shootline = []; y_shootline = [] #착탄 경로 저장
    for i in range(n_iter):
        np.random.seed(i)
        #print('iteration {}'.format(i))
        idland, xy_now, peak_z, degree, direc_now = peak_xy(cannon, balloon)
        #print('peak_z:', peak_z)
        if i == 0: 
            ax.plot( [cannon[0], balloon[0]], [cannon[1], balloon[1]], color = mycol) #cannon to balloon
    
        

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

            one_step = h_down * np.tan(degree *(np.pi / 180)) #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
            #total_move += one_step
            #print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

            wind_vec = np.array(wind_tbl.vec[idx_now-1])
            wind_vel = wind_tbl.wind_vel[idx_now-1]
            interval_now = wind_tbl.index[idx_now - 1]
            #print('고도 {} in 구간 {}에서의 바람 방향: {}'.format(wind_h[idx_now], interval_now, wind_vec))
            #print('원래 전진 방향: {}'.format(direc_now))
            
            cossim = cos_sim(direc_now, wind_vec)
            rbeta = np.random.beta(2,17)
            vel_power = (100 + cossim * 5 * wind_vel) / 100
            one_step = one_step * vel_power * (1 + rbeta) #전진거리 수정
            #print('풍속에 의해 수정된 전진 거리: {}'.format(one_step))
            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = rbeta * vel_power 
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

        
        
    #print('실제로 총 {}만큼 전진'.format(total_move))

    cir = sc.make_circle(zip(x_values, y_values))
    #ax.text(cir[0], cir[1], 'center') #최고점
    # ax.add_patch( patches.Circle( (cir[0], cir[1]), # (x, y)
    #                                         cir[2], # radius
    #     alpha=0.4, 
    #     facecolor=mycol, 
    #     edgecolor=mycol, 
    #     linewidth=2, 
    #     linestyle='solid'))
  
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
            vel_power = (100 + cossim * 100 * wind_vel) / 100
            one_step = one_step * vel_power #전진거리 수정

            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = np.random.beta(2,20) * vel_power 
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

def peak_xy(start_xyz, target_xyz):

    start_xy = start_xyz[:2]
    target_xy = target_xyz[:2]
    shoot_direc = (target_xy - start_xy) / la.norm(target_xy - start_xy); #print('shoot_direc', shoot_direc)

    h_small = target_xyz[2] - start_xyz[2]
    d_small = la.norm(start_xy - target_xy); #print('d_small:', d_small)
    side = la.norm(target_xyz - start_xyz)
    degree = np.arccos(d_small / side) * (180 / np.pi); #print('degree:', degree)

    h_big, d_final = theta2hd(degree); #print('h_big:', h_big)
    d_big = d_small * (h_big / h_small); #print('d_big:', d_big)
    
    peak_xy = start_xy + d_big * shoot_direc
    peak_z = start_xyz[2] + h_big

    ideal_landing = start_xy + d_final * shoot_direc; #print('ideal_landing:', ideal_landing)
    return ideal_landing, peak_xy, peak_z, degree, shoot_direc
    # d1 = la.norm(start_xy - target_xy) #= d_small
    # s = np.sqrt(d1**2 + h**2)
    # d2 = ran * (d1/s)
    # peak_z = ran * (h/s)

    # xy_shftd = target_xy - start_xy #shoot_direc
    # cos_and_sin = xy_shftd / d1
    # peak_xy_shftd = d2 * cos_and_sin
    # peak_xy = peak_xy_shftd + start_xy

    # theta = np.arccos(h/s)
    # direc = xy_shftd / la.norm(xy_shftd)

    # ideal_landing = peak_xy + peak_xy_shftd
    #print('바람이 없을 시 예상 착륙점: {}'.format(ideal_landing))
    #print('총 이동 거리: {}에서 0까지 내려가는 동안 {}\n'.format(peak_z, peak_z * np.tan(theta)))
    #return ideal_landing, peak_xy, peak_z, theta, direc


def cos_sim(v1, v2):
    return (np.dot(v1,v2) / (la.norm(v1) * la.norm(v2))).round(7)

def theta2hd(degree):
    #output: 높이, 거리
    if degree == 45:
        return 5000, 8000
    elif degree > 45:
        return 3200 + (40 * degree), 16000 - (degree * 1600 / 9)
    elif degree < 45:
        return 1000 / 9 * degree, 4250 + (750 / 9 * degree)



















