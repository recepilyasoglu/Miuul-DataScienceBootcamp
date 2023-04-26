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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


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


#########################################
# BONUS: Principal Component Regression
#########################################

# bileşenleri indirgeyip üzerlerine regresyon modeli kurmak

df = pd.read_csv("Machine-Learning/Datasets/hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"] and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]
len(others)

# çevirdiğimiz 3 bileşeni, dataframe'e çevirip, isimlendirdik
pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]),
                      df[others]], axis=1)
final_df.head()  # 16 değişken vardı, şuan da 3 değişken var, bu 3 değişken o 16 değişkenin %82'sini temsil ediyor


# one hot encode da kullanılabilirdi
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]  # bağımlı değişken
X = final_df.drop(["Salary"], axis=1)  # bağımsız değişken

lm = LinearRegression()

lm_rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
lm_rmse
# 345.6021106351967
y.mean()
# 535.9258821292775
# -> kötü değil çok iyi de değil ama şimdilik gayet iyi

cart = DecisionTreeRegressor()

cart_rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
cart_rmse
# 381.1688506781601

# hiperparametre optimizasyonu yapıyoruz

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_best_grid.best_params_

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 330.1964109339104

# Mülakat Sorusu :
# Elimde bir veri seti var ama veri setinde label yok ama sınıflandırma problemi çözmek/sınıflandırma modeli kurmak istiyorum ne ypabilirim ?
# Cevap:
# örneğin 1000 tane müşteri cluster'ladık ama yeni bir müşteri geldi napıcaz ?
# önce unsupervised şeklinde çeşitli cluster'lar çıkarırım sonra bu cluster'lar = sınıflar diye düşünürüm (4 cluster = 4 sınıf gibi),
# sorna bunu sınıflandırıcıya sokarım, yeni bir müşteri geldiğinde elimdeki sınıfalardan hangi birine ait  olduğunu öğrenebilirim

# Kısaca : önce unsupervised bir yöntem kullanırım, burdan çıkaracağım cluster'lara label muamelesi yaparım,
# daha sonra bunu bir sınıflandırıcıya sokup eni bir gözlem birimi vs. gediğinde bunu sınıflandırabilirim.


#############################################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
#############################################################

#################
# Breast Cancer
#################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("Machine-Learning/Datasets/breast_cancer.csv")
df.head()
df.shape

y = df["diagnosis"]  # bağımlı değişken
X = df.drop(["diagnosis", "id"], axis=1)  # bağımsız değişkenler

# 2 bileşene/boyuta indirgeme işlemi
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)


# target değişkenin eşsiz sınıflarını bulacak, bunlar liste halinde tutacak,
# her bir target'a göre seçin işlemini yapıp iki boyuta göre scatter plot oluşturucak
# ve bağımlı değişkenleri grafiğin üzerine işaretliyor olacak
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


# farklı veri setleri üzerinde uygulamak istersek

################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")

################################
# Diabetes
################################

df = pd.read_csv("Machine-Learning/Datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)  # bağımsız değişkenler içerisinde kategorik değişkenler olmaması lazım

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")
# sınıfları birbirinden daha zor ayrılıyor, iç içe geçme durumu var
