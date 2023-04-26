################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df = pd.read_csv("Machine-Learning/Datasets/USArrests.csv", index_col=0)
df.head()
df.isnull().sum()
df.info()
df.dtypes
df.describe().T
df.shape

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]  # numpy array'i olduğu için

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters  # küme sayısı
kmeans.cluster_centers_  # küme merkezleri
kmeans.labels_  # küme etiketleri
kmeans.inertia_  # SSD, SSE veya SSR değeri (kütüphane de SSD olarak ifade edilmiş)

######################################
# Optimum Küme Sayısının Belirlenmesi
######################################

kmeans = KMeans()
ssd = []

K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Verisetini kümelere ayırırken, ayırmamız gereken optimumu noktayı verir
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_  # optimum küme sayımızı görmek istersek

####################################
# Final Cluster'ların Oluşturulması
####################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("Machine-Learning/Datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans  # eyaletlerin yanına hangi cluster'dan olduğu bilgisini gir
df.head()

df["cluster"] = df["cluster"] + 1
df.head()

df[df["cluster"] == 1]

df[df["cluster"] == 5]

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("cluster.csv")

################################
# Hierarchical Clustering
################################

df = pd.read_csv("Machine-Learning/Datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")  # öklid uzaklığına göre gözlem birimlerini kümelere ayırıyor
hc_average

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

# daha yalın ve sade hali
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################

# iki aday noktamıza göre çizgi çekiyoruz
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=6, linkage="average")
clusters = cluster.fit_predict(df)

df = pd.read_csv("Machine-Learning/Datasets/USArrests.csv", index_col=0)

df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = clusters_kmeans


def get_same_cluster(dataframe, cluster_min, cluster_max, hi_cluster_no, kmeans_cluster_no,):
    for i in range(cluster_min, cluster_max):
        print("########## hi_cluster_no ve kmeans_cluster_no", i, "olan gözlemler ##########", "\n", \
              dataframe[(dataframe[hi_cluster_no] == i) & (dataframe[kmeans_cluster_no] == i)])

get_same_cluster(df, 1, 7, "hi_cluster_no", "kmeans_cluster_no")


################################
# Principal Component Analysis
################################

df = pd.read_csv("Machine-Learning/Datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and "Salary" not in col]
df[num_cols].head()
df[num_cols].dtypes

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)
pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_  # bileşenlerin açıkladıkları varyans oranları (bilgi oranı)

# peş peşe iki değişkenin, 3 değişkenin, ... açıklayacak olduğu varyans nedir ?
np.cumsum(pca.explained_variance_ratio_)
# -> pca'in oluşturduğu 16 adet yeni bileşenin, hepsinin açıkladığı varyans oranı
# -> mesela 5. değişkene geldiğimizde verinin içince bulunan bilginin/varyansın %91'nin açıklandğını görüyoruz


################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_  # değişkenlerin tek başlarına bilginin ne kadarını oluşturduklarını
np.cumsum(pca.explained_variance_ratio_)  # bir ara da ne kadarını oluşturdukları bilgisini









