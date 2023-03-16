import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# gridspec inside gridspec
fig = plt.figure()
gs0 = gridspec.GridSpec(1, 2, figure=fig)
gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0])
ax1 = fig.add_subplot(gs00[0])
ax2 = fig.add_subplot(gs00[1], projection='polar')
ax3 = fig.add_subplot(gs0[1])
ax3.set_aspect('equal')

# make some dummy data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
ax3.plot(x, y)
plt.show()

