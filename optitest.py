import matplotlib.pyplot as plt
import numpy as np
import shoot_balloon as sb

x_val = np.linspace(0.001, 90 - 0.001, 100)
y_val1 = [sb.theta2hd(x)[0] for x in x_val] #h
y_val2 = [sb.theta2hd(x)[1] for x in x_val] #d

plt.plot(x_val, y_val1)
plt.plot(x_val, y_val2)
plt.show()