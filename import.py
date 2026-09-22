
#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
color_pal = sns.color_palette(palette='Pastel2')
from itertools import cycle

import statsmodels
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import warnings 
warnings.filterwarnings('ignore')

import xgboost as xgb


#%%
