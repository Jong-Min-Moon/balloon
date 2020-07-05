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

def vec2mil(v):
    radian = np.arctan2(v[1], v[0])
    if radian < 0:
        radian += 2 * np.pi 
    
    if radian <= np.pi / 2:
        radian = np.pi / 2 - radian
    else:
        radian = np.pi / 2 + (2 * np.pi - radian)
    
    return int(round(radian * 180 / np.pi * 6400 / 360, 0))
    

col_list = ['navy', 'royalblue', 'mediumblue', 'slateblue', 'darkblue' , 'orangered', 'tab:brown', 'tab:pink']
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




    return match


    
        


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



















