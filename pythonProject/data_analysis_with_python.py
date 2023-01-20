# NumPy

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

# with numpy
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)  # 0 ile 10 arasında rastgele 10 integer
np.random.normal(10, 4, (3, 4))  # ortalaması 10, standart sapması 4 olan, 3'e 4 array oluşturma

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

# reshaping
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

# Index Selection
import numpy as np

a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))
m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 999

m[2, 3] = 2.9

m[:, 0]
m[1, :]
m[0:2, 0:3]

# Fancy Index
import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3]

v[catch]

# Conditions on Numpy
import numpy as np

v = np.array([1, 2, 3, 4, 5])

# with classic loop
ab = []
for i in v:
    if i < 3:
        ab.append(i)

# with Numpy
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

# Mathematical Operations
import numpy as np

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.max(v)
np.min(v)
np.sum(v)
np.var(v)

# İki bilinmeyenli denklem çözümü
# 5+x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

# Pandas
import pandas as pd

s = pd.Series([18, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

# Reading Data
import pandas as pd
import seaborn as sns

df = pd.read_csv("Advertising.csv")
df.head()

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()  # is there any miss value? (even if there is one)
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

# Selection in Pandas
df.index
df[0:13]
df.drop(0, axis=0).head()

delete_index = [1, 3, 5, 7]
df.drop(delete_index, axis=0).head(10)

# Converting Value to Index
df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)  # with inplace
df.index

# Converting Index to Value

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

# second way
df.reset_index().head()
df = df.reset_index().head()
df.head()

# Operations on Variables
import pandas as pd
import seaborn as sns

pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df  # does this column is exist in dataframe?

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

# it comes as a dataframe when square brackets are used twice.
# and comes as pandasseries once square brackets are used
type(df[["age"]].head())

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
df

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head()  # selected values that contain age
df.loc[:, ~df.columns.str.contains("age")].head()  # not included

# iloc & loc
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3]
df.iloc[0, 0]

# loc: label based selection
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

# Condition Selection
import pandas as pd
import seaborn as sns

pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

# parentheses are used for more than one condition
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

# Aggregation & Grouping
import pandas as pd
import seaborn as sns

pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                                        "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean", "sum"],
    "survived": "mean",
    "sex": "count"})

# Pivot Table
import pandas as pd
import seaborn as sns

pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

# gender survival rates by age distribution

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

# converting categorical to integer or or vice versa
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

# parameter: intersection, row index, column index
df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])  # plus class

# Apply & Lambda
import pandas as pd
import seaborn as sns

pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

# age variables divided by 10
df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

# normally
(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()

# with function
for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df.head()

# with apply & lambda
df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

# Join Operations
import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], ignore_index=True)

# join operatons with Merge
df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)  # already default merge by employees
pd.merge(df1, df2, on="employees")  # merge by employees

# Purpose: we want to reach the information of each employee's manager.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                    "manager": ["Caner", "Mustafa", "Berkcan"]})

df5 = pd.merge(df3, df4, on="group")

# Data Visualization: Matplotlib & Seaborn

# Categorical variables: column chart, countplot bar
# Numeric variables: hist, boxplot

# Visualization of Categorical Variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

# Visualization of Numerical Variables
plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

# plot
import numpy as np

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

# marker
y = np.array([13, 28, 11, 100])

plt.plot(y, marker="o")
plt.show()

plt.plot(y, marker="*")
plt.show()

# Line
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")
plt.show()

# Multiple Lines
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

# Labels
x = np.array([80, 85, 90, 95, 100])
y = np.array([240, 250, 260, 270, 280])
plt.plot(x, y)
plt.title("Main Title")

#name of x axis
plt.xlabel("Name of X axis")
#name of y axis
plt.ylabel("Name of Y axis")

plt.grid()
plt.show()

#Subplots
#plot1
x = np.array([80, 85, 90, 95, 100])
y = np.array([240, 250, 260, 270, 280])
plt.subplot(1, 2, 1) #meaning: i'm creating a one row two column chart and now i'm creating the first graph
plt.title("1")
plt.plot(x, y)

x = np.array([8, 8, 9, 9, 10])
y = np.array([24, 20, 26, 27, 28])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()


# Seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")
df.head()

# Visualization of Categorical Variables
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

# Visualization of Numerical Variables
sns.boxplot(x=df["total_bill"])
plt.show()

#pandas
df["total_bill"].hist()
plt.show()

