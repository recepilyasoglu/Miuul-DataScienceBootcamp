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


import warnings
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Qt5Agg")
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

# Task 1: Preparing the Data

# Adım 1: flo_data_20K.csv verisini okutunuz.

data = pd.read_csv("Miuul-DataScienceBootcamp/Tasks/10.hafta/flo_data_20k.csv")

df = data.copy()
df.head()
df.isnull().sum()
df.describe().T

df["order_channel"].value_counts()


# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

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

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Tip düzenlemeleri

df[cat_cols].dtypes
df[num_cols].dtypes
df[cat_but_car].dtypes

# interested_in_categories_12 değişkeni kategorik olduğu cat_cols'a ekledim.
cat_cols.append(cat_but_car[5])
cat_but_car = cat_but_car[1:5]

# tarihleri date formatına çevirdim
df[cat_but_car] = df[cat_but_car].apply(pd.to_datetime)


# Preparing rfm, monetary and tenure variables
def prep_rfm_metrics(dataframe, csv=False):
    # Verinin Hazırlanması
    dataframe["total_number_purchase"] = dataframe["order_num_total_ever_offline"] + dataframe[
        "order_num_total_ever_online"]
    dataframe["total_number_price"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

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

# oluşturduğum rfm_df ve df birleştirme işlemi
main_df = pd.merge(df, rfm_df, on="master_id")

main_df

# eksik değerimiz yok onun için herhangi bir işlem yapmıyorum
main_df.isnull().sum()


# One Hot Encoding
ohe_cols = [col for col in main_df.columns if 10 >= main_df[col].nunique() > 2]


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


main_df = one_hot_encoder(main_df, ohe_cols)
main_df.head()

# scaler esnasında id ve date değişkenlerinde tip hatası verme ihtimaline karşılık
# ilili değişkenleri düşürüyorum
main_df = main_df.drop(["master_id", "first_order_date", "last_order_date",
                        "last_order_date_online", "last_order_date_offline", "interested_in_categories_12"], axis=1)
main_df

# Görev 2: Customer Segmentation with K-Means (K-Means ile Müşteri Segmentasyonu)

# Adım 1: Değişkenleri standartlaştırınız.

# standartlaştırma işleminde oluşturduğum rfm_df üzerinden ilerlemeyi tercih ettim

sc = MinMaxScaler((0, 1))
scaled_df = sc.fit_transform(main_df)

scaled_df

kmeans = KMeans(n_clusters=4, random_state=17).fit(scaled_df)
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
    kmeans = KMeans(n_clusters=k).fit(scaled_df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Verisetini kümelere ayırırken, ayırmamız gereken optimumu noktayı verir
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(scaled_df)
elbow.show()

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

# Final Cluster'ların Oluşturulması

kmeans = KMeans(elbow.elbow_value_).fit(scaled_df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

clusters_kmeans = kmeans.labels_

# oluşturduğum rfm_df'in kopyası üzerinde kmeleme merkezlerini oluşturmayı düşündüm
cluster_df = main_df.copy()

cluster_df["kmeans_cluster_no"] = clusters_kmeans

cluster_df

# cluster = 0 görmek istemediğimden +1 ekledim
cluster_df["kmeans_cluster_no"] = cluster_df["kmeans_cluster_no"] + 1

cluster_df

cluster_df[cluster_df["kmeans_cluster_no"] == 1]

cluster_df[cluster_df["kmeans_cluster_no"] == 5]

cluster_df.describe().T

# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

cluster_df.groupby("kmeans_cluster_no").agg(["count", "mean", "median"])


# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

scaled_df

hc_average = linkage(scaled_df, "average")  # öklid uzaklığına göre gözlem birimlerini kümelere ayırma
hc_average

# plt.figure(figsize=(10, 5))
# plt.title("Hiyerarşik Kümeleme Dendogramı")
# plt.xlabel("Gözlem Birimleri")
# plt.ylabel("Uzaklıklar")
# dendrogram(hc_average,
#            leaf_font_size=10)
# plt.show()

# optimum küme sayısını belirleme
# iki aday noktamıza göre çizgi çekiyoruz
plt.figure(figsize=(12, 10))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=1.5, color='r', linestyle='--')
plt.axhline(y=1.8, color='b', linestyle='--')
plt.show()

# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

cluster = AgglomerativeClustering(n_clusters=18, linkage="average")
clusters = cluster.fit_predict(scaled_df)

cluster_df["hi_cluster_no"] = clusters

cluster_df["hi_cluster_no"] = cluster_df["hi_cluster_no"] + 1

cluster_df

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

cluster_df.groupby("hi_cluster_no").agg(["count", "mean", "median"])

cluster_df.groupby("hierarchical_cluster_name").agg(["count", "mean", "median"])

cluster_df


# BONUS
# benzer cluster'lara sahip gözlemleri görebilmek ve incelemek için fonksiyon yazdım
def get_same_cluster(dataframe, cluster_min, cluster_max, hi_cluster_no, kmeans_cluster_no):
    for i in range(cluster_min, cluster_max):
        print("########## hi_cluster_no ve kmeans_cluster_no", i, "olan gözlemler ##########", "\n", \
              dataframe[(dataframe[hi_cluster_no] == i) & (dataframe[kmeans_cluster_no] == i)])

pd.crosstab(cluster_df["kmeans_cluster_no"], cluster_df["hi_cluster_no"])

get_same_cluster(cluster_df, 1, 18, "kmeans_cluster_no", "hi_cluster_no")


# burda da iki farklı cluster da aynı segmentlere sahip gözlemleri incelemek istedim,
# fark olarak kullanıcının görmek istediği segmenet senaryosuna göre ilerledim
# def get_same_cluster_name(dataframe, kmeans_name, hi_name):
#     cluster_names = {1: "Hibernating",
#                      2: "At_Risk",
#                      3: "About_to_Sleep",
#                      4: "Can't_Loose",
#                      5: "Big_Spenders",
#                      6: "Loyal_Customers"}
#     print("Cluster Names:", cluster_names)
#     segment = str(input("Görmek istediğiniz müşteri segmentini belirtiniz..:"))
#     return print("########## K-Means Segment Name ve Hierarchical Segment Name", segment, "olan gözlemler ##########", "\n",
#                  dataframe[(dataframe[kmeans_name] == segment) & (dataframe[hi_name] == segment)])
#
# get_same_cluster_name(cluster_df, "kmeans_cluster_name", "hierarchical_cluster_name")

