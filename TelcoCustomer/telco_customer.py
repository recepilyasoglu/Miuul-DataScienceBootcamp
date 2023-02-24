# 1- Import the required libraries. And then Telco Cutomer Read the Churn dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Telco-Customer-Churn.csv")
df = data.copy()
df.head()

# 2- Telco Customer Churn data set;
#• Shape
#• Dtypes
#• Head, Tail
#• Missing Value
#• Describe
#obtain information.

df.shape
df.dtypes
df.head()
df.tail()
df.isnull().sum()
df.describe().T


# 3-Navigating in the Gender column of the dataset and "Male" of the gender column
#It prints 0 when it encounters the class, 1 when it encounters the opposite situation.
#Type comphrension and create a new file named "num_gender". assign to variable

#def check_gender(gender):
#        if gender == "Male":
#            return 0
#        else:
#            return 1

#df["num_gender"] = df["gender"].apply(lambda x: check_gender(x))

df["num_gender"] = [0 if col == "Male" else 1 for col in df.gender]

