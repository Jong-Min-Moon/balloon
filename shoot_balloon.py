import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations
import smallestenclosingcircle as sc

enemy_col_list = ['indianred', 'firebrick', 'maroon', 'red', 'crimson' , 'orangered', 'tab:brown', 'tab:pink'] #그래프 그릴 때 사용할 색상의 목록.
bullet_col_list = ['#F9A602', '#FCE205', '#F8DE7E', '#C49102', '#FCD12A']

def rotate_matrix(theta):
    #2차원 벡터를 시계 반대 방향으로 theta(radian)만큼 회전시키는 함수. 바람 벡터를 만들 때 사용.
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
wind_dir = pd.DataFrame({'wind_dir': [0] + list(range(1, 6400)[::-1]) ,'vec': [np.dot(rotate_matrix( i * np.pi / 3200), np.array([0,1]))for i in range(6400)] })#남풍을 0으로 하여 바람 방향을 6400개로 나누고 각 방향의 unit vector를 저장



def cos_sim(v1, v2):
    #포탄의 진행 방향 벡터와 현재 고도의 바람 방향 벡터 사이의 차이(코사인 유사도)을 측정하는 함수
    #input: 2차원 array, 2차원 array
    #output: -1에서 1 사이의 실수(코사인 유사도)
    return (np.dot(v1,v2) / (la.norm(v1) * la.norm(v2))).round(7)


def theta2hd(degree):
    #고사포의 발사각(degree)에 따라 포탄의 최대 높이(미터)와 발사 거리(미터)를 계산하는 함수
    #input: 0에서 90 사이의 실수(degree)
    #output: 최대 높이(미터), 발사 거리(미터)
    if degree == 45:
        return 5000, 8000
    elif degree > 45:
        return 3200 + (40 * degree), 16000 - (degree * 1600 / 9)
    elif degree < 45:
        return 1000 / 9 * degree, 4250 + (750 / 9 * degree)

def five_digits(v):
    fd = ''
    for num_str in v.round().astype(int).astype(str):

        if len(num_str) < 5:
            zeros = ''.join(['0' for _ in range(5 - len(num_str))])
            fd = fd + zeros + num_str
        else:
            fd = fd + num_str
        fd = fd + ' '
    return fd


def vec2mil(v):
    #2차원 벡터를 mil각도로 변환하는 함수
    #input:
    #output:
    radian = np.arctan2(v[1], v[0])
    if radian < 0:
        radian += 2 * np.pi 
    
    if radian <= np.pi / 2:
        radian = np.pi / 2 - radian
    else:
        radian = np.pi / 2 + (2 * np.pi - radian)
    
    return int(round(radian * 180 / np.pi * 6400 / 360, 0))


def allocate(cannons, balloons):
    #각 고사포에게 가장 가까운 풍선을 할당해 주는 함수
    #input: 행이 고사포 개수, 열이 고사포 좌표(x, y, z)인 pandas DataFrame
    #output: i번째 값이 i번째 고사포에 할당된 풍선을 의미하는 tuple
    m = len(cannons); n = len(balloons)
    dist_mat = np.zeros((m, n)) #거리를 저장할 행렬
    
    #각 고사포에서 각 풍선까지의 거리 계산
    for i in range(m):
        for j in range(n):
            a = np.array(cannons.iloc[i, :])
            b = np.array(balloons.iloc[j, :])
            dist_mat[i,j] = la.norm(a - b)  
    allocation = []
    #각 고사포마다 가장 가까운 풍선을 선택
    for i in range(m):
        distances = dist_mat[i,:]
        this_alloc = distances.argmin()
        allocation.append(this_alloc)

    return allocation

def allocate2(cannons, balloons):
    #각 고사포에게 가장 가까운 풍선을 할당해 주는 함수
    #input: 행이 고사포 개수, 열이 고사포 좌표(x, y, z)인 pandas DataFrame
    #output: i번째 값이 i번째 고사포에 할당된 풍선을 의미하는 tuple
    m = len(cannons); n = len(balloons)
    dist_mat = np.zeros((m, n)) #거리를 저장할 행렬
    
    #각 고사포에서 각 풍선까지의 거리 계산
    for i in range(m):
        for j in range(n):
            a = np.array(cannons.iloc[i, :])
            b = np.array(balloons.iloc[j, :])
            dist_mat[i,j] = la.norm(a - b)  
    allocation = []
    available = []
    #각 고사포마다 가장 가까운 풍선을 선택
    for i in range(m):
        distances = dist_mat[i,:]
        print(distances)
        this_alloc = distances.argmin()
        this_avail = np.where( (distances <= 3150) == 1)[0]
        print(this_avail)

        allocation.append(this_alloc)
        available.append(this_avail)

    return available


def peak_xy(start_xyz, target_xyz):
    #발사 지점의 좌표와 풍선의 좌표를 이용해 바람의 영향이 없을 때의 고사포의 낙탄 지점과 포탄 최고점의 좌표, 발사각, 발사 방향을 계산하는 함수
    #input: 3차원 array, 3차원 array
    #output: 2차원 array(낙탄 지점의 x,y 좌표. 고도는 발사 지점의 고도와 같다고 가정),
    #        2차원 array(포탄 최고점의 x,y 좌표),
    #        2차원 array(포탄 최고점의 z 좌표),
    #        0에서 90 사이의 실수(발사각(상하)),
    #        2차원 array(발사 방향(좌우))
    
    #발사 방향(좌우) 계산
    start_xy = start_xyz[:2]
    target_xy = target_xyz[:2]
    shoot_direc = (target_xy - start_xy) / la.norm(target_xy - start_xy)

    #발사각(상하) 계산. 발사점과 풍선이 만드는 직각삼각형이 발사점과 포탄 최고점이 만드는 직각삼각형과 닮음임을 이용
    h_small = target_xyz[2] - start_xyz[2]
    d_small = la.norm(start_xy - target_xy)
    side = la.norm(target_xyz - start_xyz)
    degree = np.arccos(d_small / side) * (180 / np.pi)

    #발사점에서 포탄최고점까지의 지면상의 거리 및 포탄 최고점의 높이를 계산
    h_big, d_final = theta2hd(degree) 
    d_big = d_small * (h_big / h_small)
    
    #발사 방향과 위에서 계산한 지면상의 거리를 이용해 포탄 최고점의 좌표를 계산
    peak_xy = start_xy + d_big * shoot_direc
    peak_z = start_xyz[2] + h_big

    #낙탄점의 xy 좌표를 계산
    ideal_landing = start_xy + d_final * shoot_direc
   
    #바람의 영향을 받는 거리
    d_wind = d_final - d_big
    return ideal_landing, peak_xy, peak_z, degree, shoot_direc, d_wind



def shoot(n_iter, cannon, balloon, wind_tbl, ax, col_id):
    #고사포에서 풍선을 향해 포탄을 여러 발 발사했을 때 바람의 영향을 고려한 낙탄점들과 그 중심점을 계산하는 함수
    #input: 1이상의 정수(발사 횟수),
    #       3차원 array(고사포의 좌표),
    #       3차원 array(풍선의 좌표),
    #       행이 고도, 열이 바람의 방향 벡터와 풍속인 DataFrame,
    #       matplotlib 객체(그래프를 그리는 공간),
    #       0에서 7 사이의 정수(색상 코드)
    
    enemy_col = enemy_col_list[col_id] #적 고사포 색상
    bullet_col = bullet_col_list[col_id]

    ax.scatter(cannon[0], cannon[1], color = enemy_col); ax.text(cannon[0], cannon[1], 'fire') #고사포의 위치를 그래프에 기록
    ax.scatter(balloon[0], balloon[1], color = 'royalblue', s = 300); ax.text(balloon[0], balloon[1], 'balloon') #풍선의 위치를 그래프에 기록

    x_values = []; y_values = [] #낙탄지점 저장소
    for i in range(n_iter):
        np.random.seed(i) #재현가능성을 위해 random number의 seed를 발사 순서로 정함
        idland, xy_now, peak_z, degree, direc_now, d_wind = peak_xy(cannon, balloon) #고사포와 풍선의 좌표로부터 이론적 낙탄지점과 포탄 최고점, 발사각과 발사방향
        atan = d_wind / peak_z
        #print('peak_z:', peak_z)
        if i == 0:
            #ax.scatter(xy_now[0], xy_now[1], color = mycol); ax.text(xy_now[0], xy_now[1], 'peak') #최고점
            ax.scatter(idland[0], idland[1], s = 40, alpha = 0.7, color = bullet_col); ax.text(idland[0], idland[1], 'theoretical') 
            ax.plot( [cannon[0], balloon[0]], [cannon[1], balloon[1]], color = bullet_col) #cannon to balloon
            ax.plot( [balloon[0], xy_now[0]], [balloon[1], xy_now[1]], color = bullet_col) #baloon to peak
            ax.plot( [xy_now[0], idland[0]], [xy_now[1], idland[1]], linestyle = '--', color = bullet_col) #peak to ideal landing
        

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

            one_step_old = h_down * atan #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
            #total_move += one_step
            #print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

            wind_vec = np.array(wind_tbl.vec[idx_now-1])
            wind_vel = wind_tbl.wind_vel[idx_now-1]
            interval_now = wind_tbl.index[idx_now - 1]
            #print('고도 {} in 구간 {}에서의 바람 방향: {}'.format(wind_h[idx_now], interval_now, wind_vec))
            #print('원래 전진 방향: {}'.format(direc_now))
            
            cossim = cos_sim(direc_now, wind_vec)
            rbeta = np.random.beta(3,8)
            #print('rbeta:', rbeta)
            one_step = one_step_old * ( 1 + (wind_vel / 60 * cossim * (1 + rbeta)) )  #전진거리 수정
            #print('풍속에 의해 수정된 전진 거리: {}'.format(one_step))
            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = ( 1 + (wind_vel / 60 * cossim) )  * rbeta / 4
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

        
        ax.scatter(xy_now[0], xy_now[1], s = 0.5 , color = bullet_col) #착탄지점 그래프에 그리기
        
    #print('실제로 총 {}만큼 전진'.format(total_move))

    cir = sc.make_circle(zip(x_values, y_values))
    ax.text(cir[0], cir[1], 'center') #최고점
    ax.add_patch( patches.Circle( (cir[0], cir[1]), # (x, y)
                                            cir[2], # radius
        alpha=0.35, 
        facecolor=bullet_col, 
        linewidth=2, 
        linestyle='solid'))
  
    return idland, np.array([cir[0], cir[1]])

def optim(n_points, degree, v, cannon, enemy, wind_tbl):
    center = shoot_for_optim(n_points, cannon, enemy, degree, v, wind_tbl)
    return la.norm(center - enemy[:2])

    

def drawplot(n_iter, cannons, balloons, winds, ax):
    
    wind_tbl = pd.merge(winds, wind_dir, how = 'left', on = 'wind_dir').set_index(winds.index)
    print(wind_tbl)

    idland = []; actland = []
    per = allocate(cannons, balloons)
    #print(list(zip(range(len(cannons)), per)))
    for i, comb in enumerate(zip(range(len(cannons)), per)):
        # print('{}번째 발사. 포탄과 풍선 조합: {}'.format(i, comb))
        # print('1', cannons.iloc[comb[0], :])
        # print('2', balloons.iloc[comb[1], :])
        this_idland, this_actland = shoot( n_iter,  np.array(cannons.iloc[comb[0], :]), np.array(balloons.iloc[comb[1], :]), wind_tbl,  ax, i)
        idland.append(this_idland); actland.append(this_actland)
    return per, idland, actland





def shoot_for_optim(n_points, cannon, enemy, degree, direc, wind_tbl):

    x_values = []; y_values = [] #착탄점들의 중심점을 구하기 위해 다 저장

    h, d_final = theta2hd(degree)
    #print('d_final:', d_final)
    idland_og = cannon[:2] + d_final * direc
    #print('idland:', idland_og)
    d = h / np.tan(degree * (np.pi / 180))
    
    peak_z_og = cannon[2] + h
    xy_now_og = cannon[:2] + d * direc
    for i in range(n_points):
        idland = idland_og
        xy_now = xy_now_og
        peak_z = peak_z_og
        direc_now = direc
        np.random.seed(i)

        #initialize



        # shoot
        #현재 고도보다 바로 위에 있는 테이블 값의 풍향을 사용할 것임
        wind_h = [idx[1] for idx in wind_tbl.index]; wind_h.insert(0,0)
        idx_now = (wind_h >= peak_z).tolist().index(True)
        h_now = peak_z

        #포탄 경로 그리기 준비      
        #total_move = 0
        while idx_now > 0:
            h_down = h_now - wind_h[idx_now - 1] #수직 하강 거리


            one_step = h_down * np.tan(degree *(np.pi / 180)) #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
     

            wind_vec = np.array(wind_tbl.vec[idx_now-1])
            wind_vel = wind_tbl.wind_vel[idx_now-1]
            interval_now = wind_tbl.index[idx_now - 1]
 
            
            cossim = cos_sim(direc_now, wind_vec)
            rbeta = np.random.beta(2,40)
            
            vel_power = (100 + cossim * 1.5 * wind_vel) / 100
            one_step = one_step * vel_power * (1 + rbeta) #전진거리 수정
            #print('풍속에 의해 수정된 전진 거리: {}'.format(one_step))
            if abs(cossim) != 1:
                #풍향과 풍속을 고려하여 방향 수정
                w = rbeta * vel_power 
                direc_now = (1-w) * direc_now + w * wind_vec.astype('float64')
                direc_now =  (direc_now ) / la.norm(direc_now) #unit vector로 만들기



                
            #계산 종료.

            #포탄의 전진.
            xy_now = xy_now + direc_now * one_step

            
            idx_now -=1
            h_now = wind_h[idx_now]

        
        x_values.append(xy_now[0])
        y_values.append(xy_now[1])

                
    #print('실제로 총 {}만큼 전진'.format(total_move))

    cir = sc.make_circle(zip(x_values, y_values))
  
  
    return np.array([cir[0], cir[1]])


def draw_ideal(per, K6s, enemies, ax):     
    #아군 K-6의 위치와 적 고사포들의 위치를 입력하면 이상적인 발사 경로를 그려 주는 함수
    #적 고사포 위치 표시
    for i in range(len(enemies)):
        enemy_coord_x = enemies.iloc[i, 0]; enemy_coord_y = enemies.iloc[i, 1] #적 고사포 좌표
        ax.scatter(enemy_coord_x, enemy_coord_y, color = 'red', s = 300) # 적 고사포를 빨간색 원으로 그림에 표시
        ax.text(enemy_coord_x, enemy_coord_y, 'enemy') #적 고사포 위치에 'enemy'라고 표시
    
    #아군 K-6에서 적 고사포로 쏘는 궤적 표시
    for i, comb in enumerate(zip(range(len(K6s)), per)):
        K6 = np.array(K6s.iloc[comb[0], :])
        enemy = np.array(enemies.iloc[comb[1], :])
        
        ax.scatter(K6[0], K6[1], color = 'royalblue', s=300); ax.text(K6[0], K6[1], 'K-6') #아군 K-6 위치를 파란색 원으로 그림에 표시
        ax.plot( [K6[0], enemy[0]], [K6[1], enemy[1]], color ='royalblue') #아군 K-6에서 적 고사포까지의 궤적을 파란색 선으로 표시



















