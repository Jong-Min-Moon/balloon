import pandas as pd
from numpy import linalg as la
import numpy as np
from itertools import permutations

cannons = pd.DataFrame({'x': [240,1000,2000], 'y':[140, 150, 200], 'z' : [50, 25, 130]})
balloons = pd.DataFrame({'x': [250, 1100, 2300], 'y':[1300, 1100, 1600], 'z':[1500,1000,2000]})

#print(cannons)
#print(balloons)


def allocate(cannons, balloons):
    n = len(balloons)
    dist_mat = np.zeros((n, n))
    
    for i in cannons.index:
        for j in balloons.index:
            dist_mat[i,j] = la.norm( cannons.iloc[i, :] - balloons.iloc[j, :] )
    
    print(dist_mat)
    
    sum_min = np.inf
    permu_min = 0
    for permu in permutations(range(n), n):
        sum = 0
        for tup in (zip(range(n), permu)):
            sum += dist_mat[tup]
        print(permu, sum)
        if sum < sum_min:
            permu_min = permu
            sum_min = sum

    return permu_min, sum_min

per, summ = allocate(cannons, balloons)
print(per)
print(summ)