import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd


a = {'x':[1,2,3],'y':[1,2,3]}
data = pd.DataFrame(a)
fig1 = plt.figure()

ax1 = fig1.add_subplot(111)



# (0) scatter plot

ax1.plot('x', 'y', data=a, linestyle='none', marker='o')


# (2) adding a circle

ax1.add_patch(

    patches.Circle(

        (1.5, 0.25), # (x, y)

        0.5, # radius

        alpha=0.2, 

        facecolor="red", 

        edgecolor="black", 

        linewidth=2, 

        linestyle='solid'))


plt.show()