import numpy as np
import pandas as pd

#data

wind_dic = {'h': [0,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000],
            'dir':[1,3,2,4,4,6,7,8,3,4,2,6,1,2,3,2]
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
def normalize(vec):
    return vec / np.sqrt(np.sum(np.square(vec)))


def peak_xy(start_xy, target_xyz, ran):
    target_xy = target_xyz[:2]
    h = target_xyz[2]

    d1 = np.sqrt(np.sum(np.square(start_xy - target_xy)))
    s = np.sqrt(d1**2 + h**2)
    d2 = ran * (d1/s)
    peak_z = ran * (h/s)

    xy_shftd = target_xy - start_xy
    cos_and_sin = xy_shftd / d1
    peak_xy_shftd = d2 * cos_and_sin
    peak_xy = peak_xy_shftd + start_xy

    theta = np.arccos(h/s)
    direc = normalize(xy_shftd)
    print('바람이 없을 시 예상 착륙점: {}\n'.format(peak_xy + peak_xy_shftd))
    return peak_xy, peak_z, theta, direc



cannon = np.array([0,0])
balloon = np.array([np.sqrt(2), 1, 1] * 4000)
xy_now, h, theta, direc_now = peak_xy(cannon, balloon, 4000)


# shoot
idx_now = (wind_tbl.h >= h).tolist().index(True)
h_now = wind_tbl.h[idx_now]

while idx_now > 0:
    h_down = h_now - wind_tbl.h[idx_now - 1]
    print('고도 {} -> {}. {}만큼 하강'.format(h_now, wind_tbl.h[idx_now - 1], h_down))

    one_step = h_down * np.tan(theta)
    print('x,y좌표 기준으로 {}만큼 전진'.format(one_step))

    wind_vec = wind_tbl.vec[idx_now]
    print('고도 {}에서의 바람 방향: {}'.format(wind_tbl.h[idx_now], wind_vec))
    print('원래 전진 방향: {}'.format(direc_now))
    
    direc_now =  normalize(direc_now + wind_vec)
    print('바람에 의해 수정된 방향: {}'.format(direc_now))

    xy_now = xy_now + direc_now * one_step
    print('{},{}에 도착\n'.format(xy_now, wind_tbl.h[idx_now - 1]))
    
    idx_now -=1
    h_now = wind_tbl.h[idx_now]
