import gtsam
from gtsam.symbol_shorthand import L, X
import numpy as np
from plotting import *
import pickle

# open estimate
with open('results.pkl', 'rb') as handle:
    estimate = pickle.load(handle)

# set up figure
fig = plt.figure(figsize=(20,20))
ax_3d = fig.add_subplot(1, 1, 1, projection='3d')

plot_3d(estimate, ax_3d, 60)

plt.show()