import matplotlib.pyplot as plt
import numpy as np
import shoot_balloon as sb

fig, ax = plt.subplots()
        self.data['cannon'].iloc[0,:] = [610, 6400, 150]
        self.data['cannon'].iloc[1,:] = [3350, 8000, 200]
        self.data['cannon'].iloc[2,:] = [4900, 7700, 60]
        self.data['cannon'].iloc[3,:] = [6700, 8600, 500]
        self.data['cannon'].iloc[4,:] = [8400, 7200, 320]

        
        self.data['balloon'].iloc[0,:] = [2500, 3900, 4000]
        self.data['balloon'].iloc[1,:] = [4500, 5200, 4000]
        self.data['balloon'].iloc[2,:] = [7500, 5300, 4200]

data['wind'].iloc[0,:] = [3200, 10]
data['wind'].iloc[1,:] = [4200, 4]
data['wind'].iloc[2,:] = [3400, 6]
data['wind'].iloc[3,:] = [4000, 7]
data['wind'].iloc[4,:] = [4100, 7]
data['wind'].iloc[5,:] = [3100, 9]
data['wind'].iloc[6,:] = [4000, 6]
data['wind'].iloc[7,:] = [4300, 7]
data['wind'].iloc[8,:] = [4150, 11]
data['wind'].iloc[9,:] = [3800, 8]
data['wind'].iloc[10,:] = [3870, 4]
data['wind'].iloc[11,:] = [4010, 6]

X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
C = np.random.randint(0, 5, 100)
cmap_lst = [plt.cm.rainbow, plt.cm.Blues, plt.cm.autumn, plt.cm.RdYlGn]


ax.scatter(X, Y, c=C, cmap=cmap_lst[1])
plt.show()


