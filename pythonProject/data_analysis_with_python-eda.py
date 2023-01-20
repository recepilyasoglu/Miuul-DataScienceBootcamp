# Advanced Functional EDA

# general picture(genel resim)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
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

def check_df(dataframe, head=5):
    print("############# Shape #############")
    print(dataframe.shape)
    print("############# Types #############")
    print(dataframe.dtypes)
    print("############# Head #############")
    print(dataframe.head(head))
    print("############# Tail #############")
    print(dataframe.tail(head))
    print("############# NA #############")
    print(dataframe.isnull().sum())
    print("############# Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

#checking anoter data set
df = sns.load_dataset("flights")
check_df(df)

# Analysis of Categorical Variables
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

# is the type information of the related variable included in the list we have created after converting it to string?
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# find numeric looking but with categorical values in it
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

# categorical but immeasurable variables with multiple classes
# if the type of the variable is categorical and object and the number of unique classes is more than 20 it's not a measurable variable catch them
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

# what about numerical variables ?
[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df) # percent equivalents

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
    print("###################################")

cat_summary(df, "sex")  # Summary of the variable given

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
    print("###################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":  # because we got an error of type bool
        print("sdsfsfsfsdfsfsdfsfss")  # print and continue
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
        print(pd.DataFrame({col_name: df[col_name].value_counts(),
                            "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
        print("###################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: df[col_name].value_counts(),
                            "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
        print("###################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)



# Analysis of Numerical Variables
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

# how do i select numeric values from dataset ?
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
# if not in cat_cols, select them
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


# Capturing Variables and Generalizing Operations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()


# docstring
def graph_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numeric and categorical but cardinal variables in the data set

    Args: dataframe: dataframe variable names are the dataframe to be retrieved
    cat_th: int, float
        class threshold for values that are numeric but categorical, examp: if bigger than 10 it's categorical variable car_th: (cardinal threshold) = if bigger than 20 it's cardinal variable
    car_th: int, float
        class threshold for values that are categorical but cardinal values

    Returns:
        cat_cols: list
            Categorical variable list
        cat_but_car: list
            Categorical view cardinal variable list

    Notes
    ------
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat in cat_cols

    """
    #cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    # find numeric looking but with categorical values in it
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # for numerical variables
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = graph_col_names(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
    print("###################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# BONUS
# Purpose: finding the bool variable types in the data set and converting them to integers
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = graph_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                            "Ratio": 100 * df[col_name].value_counts() / len(dataframe)}))
    print("###################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)
