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

#with Numpy
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

#İki bilinmeyenli denklem çözümü
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

#Reading Data
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
df.isnull().values.any() #is there any miss value? (even if there is one)
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

#Selection in Pandas
df.index
df[0:13]
df.drop(0, axis=0).head()

delete_index = [1, 3, 5, 7]
df.drop(delete_index, axis=0).head(10)

#Converting Value to Index
df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True) #with inplace
df.index

#Converting Index to Value

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

#second way
df.reset_index().head()
df = df.reset_index().head()
df.head()

#Operations on Variables
import pandas as pd
import seaborn as sns
pd.set_option("display.max.columns", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df #does this column is exist in dataframe?

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

#it comes as a dataframe when square brackets are used twice.
#and comes as pandasseries once square brackets are used
type(df[["age"]].head())

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]
df

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head() #selected values that contain age
df.loc[:, ~df.columns.str.contains("age")].head() #not included

# iloc & loc
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

#iloc: integer based selection
df.iloc[0:3]
df.iloc[0,0]

#loc: label based selection
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]