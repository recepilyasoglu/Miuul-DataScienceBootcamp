import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Telco-Customer-Churn.csv")
df = data.copy()
df.head()