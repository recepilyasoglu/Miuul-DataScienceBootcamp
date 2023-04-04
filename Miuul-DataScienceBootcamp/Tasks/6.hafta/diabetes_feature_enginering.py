#####################################################
################## RECEP İLYASOĞLU ##################
#####################################################

############# Feature Engineering on Diabetes Dataset #############

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
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)


## Task 1 : Keşifçi Veri Analizi

# Step 1:  Genel resmi inceleyiniz

data = pd.read_csv("Tasks/6.hafta/diabetes.csv")
df = data.copy()
df.head()

df.shape
df.describe().T
df.isnull().sum() * df.shape[0] / 100
df.info()
df.dtypes

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
df[cat_cols].shape
df[cat_cols].dtypes
df[cat_cols].isnull().sum()
df[cat_cols].describe().T

df[num_cols].shape
df[num_cols].dtypes
df[num_cols].isnull().sum()
df[num_cols].describe().T

# Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre Numerik değişkenlerin ortalaması)

# Hedef değişkenimiz zaten outcome = kategorik değişken


# Kategorik değişkenlere göre hedef değişkenin ortalamas
# df.groupby(num_cols)[cat_cols].mean()  # saçma bi çıktı verdi

# hedef değişkene göre Numerik değişkenlerin ortalaması
df.groupby(cat_cols)[num_cols].mean()

# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)

def check_outlier(dataframe, col_name):  # q1 ve q3 'ü de biçimlendirmek istersek check_outlier'a argüman olarak girmemiz gerekir
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False   # var mı yok mu sorusuna bool dönmesi lazım (True veya False)

# for col in cat_cols:
#     print(col, check_outlier(df, col))

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    # 10 dan çok aykırı değer varsa head ile getir, çok gelmesin yani 10 tane gelsin
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:  # değilse hepsini geir azmış zaten
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# for col in num_cols:
#     grab_outliers(df, col, True)
#
# df[num_cols].head()

grab_outliers(df, "Age")
grab_outliers(df, "Glucose")

# Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

missing_values_table(df, True)


# Adım 7: Korelasyon analizi yapınız

# daha iyi sonuç alabilmek adına, yapılan gözlemden kendisini(Outcome) çıkardım
df.corr().sort_values("Outcome", ascending=False) \
    .drop("Outcome", axis=0)


## Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

# ilk olarak, aykırı değer var mı? yok mu? kontrolü
def check_outlier(dataframe, col_name):  # q1 ve q3 'ü de biçimlendirmek istersek check_outlier'a argüman olarak girmemiz gerekir
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False   # var mı yok mu sorusuna bool dönmesi lazım (True veya False)

for col in num_cols:
    print(col, check_outlier(df, col))  # numerik değişkenlerin hepsinde var


# baskılama
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

# replace_with_thresholds(df, "Age")
# replace_with_thresholds(df, "Insulin")
df.describe().T

df["Pregnancies"].describe().T
num_cols = num_cols[1:]

# for col in num_cols:
#     print(df[col].value_counts())

# 0 -> NAN
df.head(10)
df[num_cols] = df[num_cols].replace(0, np.nan)
df.head(10)


df.describe().T
df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

na_cols = missing_values_table(df)

df.groupby(cat_cols)[num_cols].mean()

for col in num_cols:
    # print(df[col])
    df[col] = df[col].fillna(df.groupby(cat_cols)[col].transform("mean"))


# Adım 2: Yeni değişkenler oluşturunuz.
df.head()
df["Age"]
df["NEW_AGE_CAT"] = pd.cut(df['Age'], \
                           bins=[df.Age.min()-1, df.Age.median(), 45, df.Age.max()], \
                           labels=["Young", "Mature", "Old"])
df[["Age", "NEW_AGE_CAT"]].head(20)

df["Number_of_Pregnancies"] = pd.cut(df["Pregnancies"], \
                                     bins=[df.Pregnancies.min(), df.Pregnancies.median(), 7, df.Pregnancies.max()], \
                                     labels=["Normal", "Much", "Extreme"], \
                                     right=False)
df[["Pregnancies", "Number_of_Pregnancies"]].head(20)

df["BMI"].head()
def calculate_bmi(col):
    if col["BMI"] < 18.5:
        return "Under"
    elif col["BMI"] >= 18.5 and col["BMI"] <= 24.9:
        return "Healthy"
    elif col["BMI"] >= 25 and col["BMI"] <= 29.9:
        return "Over"
    elif col["BMI"] >= 30:
        return "Obese"

df = df.assign(Result_of_BMI=df.apply(calculate_bmi, axis=1))
df.head()


# Adım 3: Encoding işlemlerini gerçekleştiriniz.

# yeni oluşturduğum kategorik değişkenleri gözlemlerken, teker teker value_counts'larına bakmak yerine
# fonksiyon yazmayı tercih ettim
new_variables = df[["Number_of_Pregnancies", "NEW_AGE_CAT", "Result_of_BMI"]]
def count_of_values(dataframe):
    for col in dataframe:
        print(dataframe[col].value_counts())

count_of_values(new_variables)

# One Hot Encoding
# eşsiz değer sayısı 2 den fazla olanların sayısı 10 dan küçük veya 10'a eşitse getir
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
df.head()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols

rare_analyser(df, "Outcome", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
for col in num_cols:
    print(col, check_outlier(df, col))

# num_cols da aykırı değerler çıktı baskılama yapalım
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


# standartlaştırma
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()


# Adım 5: Model oluşturunuz.
y = df["Outcome"]  #bağımlı değişken
X = df.drop(["Outcome"], axis=1)  # bağımsız değişkenler, ilgili sütunlar dışındaki değerler
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)  # modeli test seti üzerinde tahmin et,
accuracy_score(y_pred, y_test)  # bu değerleri kıyaslıyoruz

# oluşturduğum değişkenler ne alem de ? - önem sırası
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
