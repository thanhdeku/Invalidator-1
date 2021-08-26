import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
ticks = np.arange(0, 10, 0.1)
y1 = np.zeros((len(x)))
y2 = np.zeros((len(x)))
xmax = 0.0
imax = 0
for ii in range(len(x)):
    y1[ii] = 50 * ii - 5 * ii**2
    y2[ii] = 50 * ii - 7 * ii**2
    ticks[ii] = y1[ii]-y2[ii]
    if (y1[ii] < 0.0):
        xmax = x[ii]
        imax = ii
        break
ymax = max(y1)
fig = plt.figure(figsize=(8,4), dpi=300)
plt.plot(x, y1, 'g:')
plt.plot(x, y2, 'b:')
plt.ylim([0, ymax*1.05])
plt.xlim([0, xmax])
plt.fill_between(x, y1, y2, color='grey', alpha=0.3)

for ii in range(imax):
    plt.text(x[ii], ymax*0.01, ticks[ii])
plt.show()
