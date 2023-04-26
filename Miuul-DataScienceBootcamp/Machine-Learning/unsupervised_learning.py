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


clusters = kmeans.labels_

df = pd.read_csv("Machine-Learning/Datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters  # eyaletlerin yanına hangi cluster'dan olduğu bilgisini gir
df.head()

df["cluster"] = df["cluster"] + 1
df.head()

df[df["cluster"] == 1]

df[df["cluster"] == 5]

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("cluster.csv")
















