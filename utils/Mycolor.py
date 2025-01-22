# define cmap with RGB values

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class Mycolor:
    
    # green
    mycolors = np.array([
        [247,252,245],
        [229,245,249],
        [199,233,192],
        [161,217,155],
        [116,196,118],
        [65,171,93],
        [35,139,69],
        [0,109,44],
        [0,68,27]])/255

    # orange
    mycolors_2 = np.array([
        [255,245,235],
        [254,230,206],
        [253,208,162],
        [253,174,107],
        [253,141,60],
        [241,105,19],
        [217,72,1],
        [166,54,3],
        [127,39,4]])/255

    # purple
    mycolors_3 = np.array([
        [252,251,253],
        [239,237,245],
        [218,218,235],
        [188,189,220],
        [158,154,200],
        [128,125,186],
        [106,81,163],
        [84,39,143],
        [63,0,125]])/255

    cm = LinearSegmentedColormap.from_list('mycolors', mycolors[::-1], N=len(mycolors))
    cm_2 = LinearSegmentedColormap.from_list('mycolors_2', mycolors_2[::-1], N=len(mycolors_2))
    cm_3 = LinearSegmentedColormap.from_list('mycolors_3', mycolors_3[::-1], N=len(mycolors_3))
    
    cm_dp = sns.diverging_palette(220, 20, center="light", as_cmap=True)