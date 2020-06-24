import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt


#data

wind_dic = {'h': [0,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000],
            'dir':[4,8,4,8,4,8,7,8,3,4,2,6,1,2,3,2]
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

def peak_xy(start_xy, target_xyz, ran):
    target_xy = target_xyz[:2]
    h = target_xyz[2]

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
    #print('바람이 없을 시 예상 착륙점: {}\n'.format(ideal_landing))
    return ideal_landing, peak_xy, peak_z, theta, direc



cannon = np.array([0,0])
plt.scatter(cannon[0], cannon[1])
plt.text(cannon[0], cannon[1], 'fire')


balloon = np.array([np.sqrt(2), 1, 1] )* 4000
plt.scatter(balloon[0], balloon[1])
plt.text(balloon[0], balloon[1], 'balloon')
print(balloon)

for i in range(10):
    idland, xy_now, h, theta, direc_now = peak_xy(cannon, balloon, 8000)

    if i == 1:
        print('peak', xy_now)
        plt.scatter(xy_now[0], xy_now[1])
        plt.text(xy_now[0], xy_now[1], 'peak')
#plt.plot( [xy_now[0], idland[0]], [xy_now[1], idland[1]])

# shoot
    idx_now = (wind_tbl.h >= h).tolist().index(True)
    h_now = wind_tbl.h[idx_now]
#t.setheading(vec2ang(direc_now))

#x_values = [xy_now[0]]
#y_values = [xy_now[1]]


    while idx_now > 0:
        h_down = h_now - wind_tbl.h[idx_now - 1]
        #print('고도 {} -> {}. {}만큼 하강'.format(h_now, wind_tbl.h[idx_now - 1], h_down))

        one_step = h_down * np.tan(theta)
        #print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

        wind_vec = np.array(wind_tbl.vec[idx_now])
        #print('고도 {}에서의 바람 방향: {}'.format(wind_tbl.h[idx_now], wind_vec))
        #print('원래 전진 방향: {}'.format(direc_now))
        
        cossim = cos_sim(direc_now, wind_vec)
        if abs(cossim) == 1:
            print('바람의 방향이 포탄의 방향과 정확히 일치(혹은 정확히 반대)하므로, 포탄 진행방향 변화 없음')
        else:
            w = np.random.beta(2,20)
            #print(w)
            direc_now = (1-w) * direc_now + w * wind_vec.astype('float64')
            direc_now =  (direc_now ) / la.norm(direc_now)
            #print('바람에 의해 수정된 방향: {}'.format(direc_now))
        xy_now = xy_now + direc_now * one_step
        #print('{},{}에 도착\n'.format(xy_now, wind_tbl.h[idx_now - 1]))
        
        idx_now -=1
        h_now = wind_tbl.h[idx_now]
        
        #x_values.append(xy_now[0])
        #y_values.append(xy_now[1])

    plt.scatter(xy_now[0], xy_now[1])

#plt.plot(x_values, y_values)
plt.scatter(idland[0], idland[1], s = 500)
plt.show()

