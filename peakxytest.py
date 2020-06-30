import shoot_balloon as sb
import numpy as np
fire = np.array([0,0,0])
balloon = np.array([0,1000,1000])
print(sb.peak_xy(fire, balloon))