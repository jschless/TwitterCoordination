import preprocessing, trending
from config import TWITTER_DATA_DIR, TRENDS_DIR, ASSETS_DIR, FOLLOWER_DATA_DIR
import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import seaborn as sns
import pickle, os
import statsmodels.formula.api as smf
import graphviz as gr
import warnings, gzip
warnings.filterwarnings('ignore')


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['figure.figsize'] = (12.0, 8)
mpl.rc('font',**{'family': 'sans-serif', 'weight': 'bold', 'size': 14})
mpl.rc('axes',**{'titlesize': 20, 'titleweight': 'bold', 'labelsize': 16, 'labelweight': 'bold'})
mpl.rc('legend',**{'fontsize': 14})
mpl.rc('figure',**{'titlesize': 16, 'titleweight': 'bold'})
mpl.rc('lines',**{'linewidth': 2.5, 'markersize': 18, 'markeredgewidth': 0})
mpl.rc('mathtext',**{'fontset': 'custom', 'rm': 'sans:bold', 'bf': 'sans:bold', 'it': 'sans:italic', 'sf': 'sans:bold', 'default': 'it'})

# plt.rc('text',usetex=False) # [default] usetex should be False

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,sfmath} \boldmath']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors_default = prop_cycle.by_key()['color']
