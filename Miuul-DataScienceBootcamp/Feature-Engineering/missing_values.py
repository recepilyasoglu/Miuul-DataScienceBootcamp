#############################################
# Missing Values (Eksik Değerler)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# Eksik Değerlerin Yakalanması
#############################################

def load_application_train():
    data = pd.read_csv("Feature-Engineering/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("Feature-Engineering/titanic.csv")
    return data

df = load()
df.head()

df = load()
df.head()

# eksik gozlem var mı yok mu 
df.isnull().values.any()




