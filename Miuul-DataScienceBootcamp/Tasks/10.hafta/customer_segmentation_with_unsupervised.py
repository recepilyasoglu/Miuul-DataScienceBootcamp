#####################################################
################## RECEP İLYASOĞLU ##################
#####################################################

# Customer Segmentation with Unsupervised Learning


## Business Problem:
# FLO segments its customers and according to these segments
# Wants to define marketing strategies. for this as a result, the behavior of customers will be defined and this
# Groups will be formed according to clusters in behavior.
# (FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik
# olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.)


## Dataset Story:
# The dataset includes the last shopping from Flo to OmniChannel (both online and offline shoppers) in 2020 - 2021.
# It consists of information obtained from the past shopping behavior of customers who
# (Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır)


import joblib
import warnings
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import datetime as dt
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)


# Task 1: Preparing the Data

# Adım 1: flo_data_20K.csv verisini okutunuz.

data = pd.read_csv("Tasks/10.hafta/flo_data_20k.csv")

df = data.copy()
df.head()
df.isnull().sum()
df.describe().T

df["order_channel"].value_counts()


# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# Aykırı değer kontrolü
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

# sütunlar da kontrol ettim, gözüme çarpan değişkenle liste içerisine alıp baskılayabilmek adına
outlier_thresholds(df, df.columns)

values = ["customer_value_total_ever_offline", "customer_value_total_ever_online",
          "customer_value_total_ever_offline", "customer_value_total_ever_online"]

def check_outlier(dataframe, col_name):
    if dataframe[col_name].dtype != 'category':
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        col = pd.to_numeric(dataframe[col_name], errors='coerce')
        if col[(col > up_limit) | (col < low_limit)].any(axis=None):
            return True
        else:
            return False

for col in values:
    print(col, check_outlier(df, col))
    # sağlamasında da görmüş oldum yukarıdaki liste içerisinde yer alan değerler de aykırılık var

def replace_with_threshold(dataframe, variable):
    low_limit, upl_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > upl_limit), variable] = upl_limit

# yukarıda values içerisinde yer alan değerleri baskılama
for col in values:
    replace_with_threshold(df, col)

# kontrol
for col in values:
    print(col, check_outlier(df, col))  # False hepsi


# Preparing rfm, monetary and tenure variables
def prep_rfm_metrics(dataframe, csv=False):
    # Verinin Hazırlanması
    dataframe["total_number_purchase"] = dataframe["order_num_total_ever_offline"] + dataframe["order_num_total_ever_online"]
    dataframe["total_number_price"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]

    date = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date] = dataframe[date].apply(pd.to_datetime)

    # RFM Metriklerinin Hazırlanması
    today_date = dt.datetime(2021, 6, 1)
    rfm = df.groupby("master_id").agg(
        {"last_order_date": lambda x: (today_date - x.max()).days,
         "total_number_purchase": lambda x: x,
         "total_number_price": lambda x: x})
    rfm.columns = ["recency", "frequency", "monetary"]

    return rfm

rfm_df = prep_rfm_metrics(df)
rfm_df

df.describe().T

# yukarıda yeni oluşturduğum değişkenler de aykırılık olabilir
new_values = ["total_number_purchase", "total_number_price"]

# kontrol
for col in new_values:
    print(col, check_outlier(df, col))  # True

# burda da yine replace ile ufak bir kırpma işlemi yaptım
for col in new_values:
    replace_with_threshold(df, col)

df.describe().T

# oluşturduğum rfm_df ve df birleştirme işlemi
main_df = pd.merge(df, rfm_df, on="master_id")

main_df


# Görev 2: Customer Segmentation with K-Means (K-Means ile Müşteri Segmentasyonu)

# Adım 1: Değişkenleri standartlaştırınız.

# standartlaştırma işleminde oluşturduğum rfm_df üzerinden ilerlemeyi tercih ettim

sc = MinMaxScaler((0, 1))
sclaed_df = sc.fit_transform(rfm_df)

sclaed_df

kmeans = KMeans(n_clusters=4, random_state=17).fit(sclaed_df)
kmeans.get_params()

kmeans.n_clusters  # küme sayısı
kmeans.cluster_centers_  # küme merkezleri
kmeans.labels_  # küme etiketleri
kmeans.inertia_  # SSD, SSE veya SSR değeri (kütüphane de SSD olarak ifade edilmiş)


# Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []

K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(sclaed_df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Verisetini kümelere ayırırken, ayırmamız gereken optimumu noktayı verir
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(sclaed_df)
elbow.show()


# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

# Final Cluster'ların Oluşturulması

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(sclaed_df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

clusters_kmeans = kmeans.labels_

# oluşturduğum rfm_df'in kopyası üzerinde kmeleme merkezlerini oluşturmayı düşündüm
k_df = rfm_df.copy()

k_df["cluster"] = clusters_kmeans

k_df

k_df["cluster"] = k_df["cluster"] + 1

k_df

k_df[k_df["cluster"] == 1]

k_df[k_df["cluster"] == 5]


# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

k_df.groupby("cluster").agg(["count", "mean", "median"])

k_df.to_csv("cluster.csv")





