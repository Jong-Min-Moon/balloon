import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import permutations
import smallestenclosingcircle as sc 

enemy_col_list = ['indianred', 'firebrick', 'maroon', 'red', 'crimson' , 'orangered', 'tab:brown', 'tab:pink'] #그래프 그릴 때 사용할 색상의 목록.
bullet_col_list = ['#F9A602', '#FCE205', '#F8DE7E', '#F5F5DC', '#F8E473']

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
        allocation.append(distances.argmin())

    return allocation


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
    #print('이론적 전진거리:', d_final)
    return ideal_landing, peak_xy, peak_z, degree, shoot_direc, d_wind



def shoot(n_iter, cannon, balloon, wind_tbl, ax ,
        beta_1, beta_2):
    #고사포에서 풍선을 향해 포탄을 여러 발 발사했을 때 바람의 영향을 고려한 낙탄점들과 그 중심점을 계산하는 함수
    #input: 1이상의 정수(발사 횟수)
    #       3차원 array(고사포의 좌표)
    #       3차원 array(풍선의 좌표)
    #       행이 고도, 열이 바람의 방향 벡터와 풍속인 DataFrame
    #       matplotlib 객체(그래프를 그리는 공간)
    #       0에서 7 사이의 정수(색상 코드)
    
    

    x_values = []; y_values = [] #착탄점들의 중심점을 구하기 위해 다 저장
    
    
    for i in range(n_iter):
        np.random.seed(i) #재현가능성을 위해 random number의 seed를 발사 순서로 정함
        idland, xy_now, peak_z, degree, direc_now, d_wind = peak_xy(cannon, balloon) #고사포와 풍선의 좌표로부터 이론적 낙탄지점과 포탄 최고점, 발사각과 발사방향을 계산
        
        if i == 0:
            ax.scatter(idland[0], idland[1], s = 40, alpha = 0.7, color = 'magenta'); ax.text(idland[0], idland[1], 'theoretical') #이론적 낙탄점을 그래프에 표시
            ax.plot( [cannon[0], balloon[0]], [cannon[1], balloon[1]], color = 'magenta') # 고사포에서 풍선까지의 발사 경로를 그래프에 표시
            ax.plot( [balloon[0], xy_now[0]], [balloon[1], xy_now[1]], color = 'magenta') # 풍선에서 포탄 최고점까지의 경로를 그래프에 표시
            ax.plot( [xy_now[0], idland[0]], [xy_now[1], idland[1]], linestyle = '--', color = 'magenta') #포탄 최고점에서 이론적 낙탄지점까지의 경로를 그래프에 표시

        ## 발사 알고리즘 시작
        wind_h = [idx[1] for idx in wind_tbl.index]; wind_h.insert(0,0) #바람 정보 테이블에서 고도값을 빼와서 리스트로 저장.
        idx_now = (wind_h >= peak_z).tolist().index(True)# 현재 고도보다 바로 위에 있는 고도값에서부터 시작. 예)현재 고도가 1300미터이고 고도 테이블에 1200미터와 1500미터가 있으면 1500미터부터 시작.
        h_now = peak_z
        atan = d_wind / peak_z
        x_path = [xy_now[0]]; y_path = [xy_now[1]]
        total_moves = 0
        while idx_now > 0:
            h_down = h_now - wind_h[idx_now - 1] #수직 하강 거리 계산
            print('{}에서 {}까지 하강'.format(h_now, wind_h[idx_now - 1]))
            one_step_old = h_down * atan #전진 거리. 풍향의 영향이 없을 때의 방향을 기준으로 거리를 계산함
            #print('기존의 전진 거리:', one_step)

            wind_vec = np.array(wind_tbl.vec[idx_now-1]) #현 고도에서의 풍향 벡터 가져오기
            print('바람의 방향:', wind_vec)
            wind_vel = wind_tbl.wind_vel[idx_now-1] #현 고도에서의 풍속값 가져오기
            
            cossim = cos_sim(direc_now, wind_vec) #포탄의 현재 진행 방향(좌우) 벡터와 바람 방향 벡터의 코사인 유사도
            rbeta = np.random.beta(beta_1,beta_2) #바람의 영향 반영 비율을 결정하는 랜덤 넘버. 베타분포의 두 번째 모수가 첫 번째 모수보다 많이 클수록 바람의 영향이 작아짐
            vel_power = (100 + cossim * wind_vel * (1 + rbeta) )/100 #풍속의 영향력을 결정하는 수. 포탄의 진행 방향과 바람의 방향의 유사도의 절댓값이 클수록(방향이 매우 비슷하거나, 또는 거의 반대 방향일 경우) 풍속의 영향력이 커짐.
            print('rbeta:', rbeta); print('vel_power:', vel_power)
            one_step = one_step_old * vel_power   #풍속의 영향력을 반영하여 전진거리 수정
            total_moves += one_step
            #print('풍속과 풍향에 의해 수정된 전진거리:', one_step)
            
            if abs(cossim) != 1: #바람의 방향이 포탄의 진행방향과 완전히 같거나 정 반대이면, 포탄 진행방향은 변함이 없음
                print('기존의 진행 방향:', direc_now)
                w = rbeta * (100 + 60 * cossim)/100
                print('w:', w)
                direc_now = (1-w) * direc_now + w * wind_vec.astype('float64') #포탄의 현재 진행 방향을 바람의 방향 쪽으로 w만큼 수정
                direc_now =  (direc_now ) / la.norm(direc_now) #unit vector로 만들기
                print('수정된 방향:', direc_now, '\n')
            xy_now = xy_now + direc_now * one_step #위에서 구한 전진거리와 진행방향을 이용해 포탄을 전진시킴


                
            #계산 종료.

            #포탄의 전진.
            x_path.append(xy_now[0]); y_path.append(xy_now[1])
            idx_now -=1
            h_now = wind_h[idx_now]
            #ax.text(xy_now[0], xy_now[1], '{} -> {}'.format(round(one_step_old), round(one_step)))
        #발사 1회 완료. 낙탄점을 저장하고 그래프에 그리기
        x_values.append(xy_now[0]); y_values.append(xy_now[1])
        ax.scatter(xy_now[0], xy_now[1], s = 5, color = 'magenta') #착탄지점 그래프에 그리기
        for i in range(len(x_path)-1):
            color_code = (i+1) / len(x_path)
            ax.plot([x_path[i], x_path[i+1]], [y_path[i], y_path[i+1]], color = plt.cm.rainbow(color_code))

    #발사 n회 완료. 낙탄점의 중심점 구하기. 낙탄점을 모두 둘러싸는 최소크기의 원을 구하고 그 중심점을 낙탄점의 중심점으로 삼음.
    
    #print('실제로 총 {}만큼 전진'.format(total_move))
    

    ax.scatter(cannon[0], cannon[1], color = 'red'); ax.text(cannon[0], cannon[1], 'fire') #고사포의 위치를 그래프에 기록
    ax.scatter(balloon[0], balloon[1], color = 'mediumblue', s = 300); ax.text(balloon[0], balloon[1], 'balloon') #풍선의 위치를 그래프에 기록

    return idland, xy_now, d_wind, total_moves


    

def drawplot(n_iter, cannons, balloons, winds, ax, 
            beta_1, beta_2):
    
    wind_tbl = pd.merge(winds, wind_dir, how = 'left', on = 'wind_dir').set_index(winds.index)

    idland = []; actland = []
    d_wind = []; total_moves = []
    per = allocate(cannons, balloons)
    for i, comb in enumerate(zip(range(len(cannons)), per)):
        this_idland, this_actland, dw, tm = shoot( n_iter,  np.array(cannons.iloc[comb[0], :]), np.array(balloons.iloc[comb[1], :]), wind_tbl,  ax,
                 beta_1, beta_2)
        idland.append(this_idland); actland.append(this_actland); d_wind.append(dw); total_moves.append(tm)
    return per, idland, actland, d_wind, total_moves

#
























