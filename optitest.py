from scipy.optimize import minimize
import numpy as np
def f(x):
    return x[0]**2 + x[1]*2 + 1

minimum = minimize(f, x0 = np.array([1,1]), method = 'Nelder-Mead')
print(str(minimum))