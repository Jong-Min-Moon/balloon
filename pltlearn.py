import matplotlib.pyplot as plt



def plotit(ax):
    ax.plot([1,2,3])

fig1, ax1 = plt.subplots()

plotit(ax1)
plt.show()
