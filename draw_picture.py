import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pandas as pd
import numpy as np
import math

data1 = pd.read_excel('R_CIExyY.xlsx', sheet_name='R_hot', index_col=False)
data2 = pd.read_excel('R_CIExyY.xlsx', sheet_name='R_cool', index_col=False)

# change the axis of x value from 300nm to 360nm with 0.5nm step szie and 360 to 760nm with 1nm step size
x = list(np.linspace(300, 359.5, 120))+list(range(360, 761, 1))

# line 101
# delete the failed index, need to provide index list want to remove
# iternum = np.delete(list(range(data1.columns.shape[0]+1)), [5, 12, 22, 26]).tolist()
# index = np.delete(list(range(data1.columns.shape[0]+1)), 22).tolist()
# data1.columns = index
# data2.columns = index
b_vary = list(range(0, -61, -1))+list(range(-59, 1, 1))+list(range(1, 61, 1))+list(range(59, -1, -1))
iternum = np.delete(list(range(data1.columns.shape[0])), [49, 145, 146, 212, 213, 214, 215, 216]).tolist()
fig, ax = plt.subplots(2, 1)
ims = []

for i in iternum:
    b = b_vary[i]
    index = i
    if i <= 60:
        a = math.sqrt(3600 - b ** 2)
    elif 61 <= i <= 180:
        a = -math.sqrt(3600 - b ** 2)
    elif 181 <= i <= 240:
        a = math.sqrt(3600 - b ** 2)

    if i < 60:
        Cab = 360+(math.atan(b/a)/math.pi)*180
    elif i == 60:
        Cab = 270
    elif 61 <= i <=119:
        Cab = 180 + (math.atan(b/a)/math.pi)*180
    elif i == 120:
        Cab = 180
    elif 121 <= i < 180:
        Cab = 180 + (math.atan(b/a)/math.pi)*180
    elif i == 180:
        Cab = 90
    elif i > 180:
        Cab = (math.atan(b/a)/math.pi)*180

    y1 = np.array(data1[index])
    y2 = np.array(data2[index])

    # ax[0].set_title('L={}, a=0, b=0'.format(i))
    ttl = plt.text(0.5, 1.01, 'L=60, a={0}, b={1}, Cab={2}'.format(a, b, Cab),
                   horizontalalignment='center', verticalalignment='bottom', transform=ax[0].transAxes)

    # title = ax[0].text(240, 320, 'L={}, a=0, b=0'.format(i),
    #                 size=plt.rcParams["axes.titlesize"],
    #                 ha="center")
    ax[0].set_ylabel('Reflectivity')
    ax[0].set_ylim(-0.001, 0.0065)
    im1, = ax[0].plot(x, y1, 'r', lw=1)
    ax[1].set_xlabel('wavelength/nm')
    ax[1].set_ylabel('Reflectivity')
    ax[1].set_ylim(-0.1, 1.2)
    im2, = ax[1].plot(x, y2, 'b', lw=1)
    ims.append([im1, im2, ttl])


    # draw line chart
    # ax.bar(y, height=y, width=0.3) # draw bar chart
animator = ani.ArtistAnimation(fig, ims, interval=200, repeat=True, blit=False)
animator.save("test.gif", fps=10, writer='pillow')
plt.show()
