import numpy as np
import matplotlib.pyplot as plt
from plotstyle import PlotStyle 
from thunderfish.efield import efish_monopoles, epotential_meshgrid, squareroot_transform
from thunderfish.fishshapes import plot_fish

    
def plot_threewavefish(ax, s):
    maxx = 30.0
    maxy = 25.0
    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)
    fish1 = (('Alepto', 'top'), (-12, -12), (1, 0.7), 18.0, 10)
    fish2 = (('Alepto', 'top'), (14, 2), (0.6, 1), 20.0, -15)
    fish3 = (('Alepto', 'top'), (-8, 10), (1, -0.4), 16.0, -12)
    poles1 = efish_monopoles(*fish1[1:])
    poles2 = efish_monopoles(*fish2[1:])
    poles3 = efish_monopoles(*fish3[1:])
    pot = epotential_meshgrid(xx, yy, None, poles1, poles2, poles3)
    mz = 0.65
    zz = squareroot_transform(pot/200, mz)
    levels = np.linspace(-mz, mz, 16)
    ax.contourf(x, y, -zz, levels)
    ax.contour(x, y, -zz, levels, **s.csLine)
    plot_fish(ax, *fish1, bodykwargs=s.fishBody, finkwargs=s.fishFins)
    plot_fish(ax, *fish2, bodykwargs=s.fishBody, finkwargs=s.fishFins)
    plot_fish(ax, *fish3, bodykwargs=s.fishBody, finkwargs=s.fishFins)
    #ax.xscalebar(0.99, 0.06, 5, 'cm', ha='right')

    
if __name__ == "__main__":
    s = PlotStyle()
    plt.rcParams['image.cmap'] = 'RdYlBu'
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    plot_threewavefish(ax, s)
    fig.savefig()
