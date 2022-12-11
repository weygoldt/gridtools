import matplotlib.pyplot as plt
import numpy as np
import pywt

x = np.arange(512)

y = np.sin(2*np.pi*x/32)

coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')

plt.matshow(coef) 

plt.show()