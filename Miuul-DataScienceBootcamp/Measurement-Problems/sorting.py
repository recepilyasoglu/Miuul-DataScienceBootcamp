### Sorting Products

# Uygulama: Kurs Sıralama
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("Measurement-Problems/Datasets/product_sorting.csv")
df.head()
df.shape


# Sorting by Rating
df.sort_values("rating", ascending=False).head(20)


# Sorting by Comment Count or Purchase Count
df.sort_values('purchase_count', ascending=False).head(20)

df.sort_values('commment_count', ascending=False).head(20)


# Sorting by Rating, Comment and Purchase

#oluşturacağım değer 1 ve 5 arasında olacak sonra da fit etmemizi istiyor onu da yaptık
#sonra bu dönüştürmeyi yağtığında eski purchase_count'lar var bir de yeni 1 ve 5 arasında
#olan purchase_count'lar var bekliyor MinMaxScaler, bekleme ve bu işlemi tamamla, transform et ve
#yeni değiştirdiğin değerlere  bunu dönüştür diyoruz
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[['purchase_count']]). \
    transform(df[['purchase_count']])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[['commment_count']]). \
    transform(df[['commment_count']])

#skor
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42 ):
    return (df["comment_count_scaled"] * w1 / 100 +
            df["purchase_count_scaled"] * w2 / 100 +
            df["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values('weighted_sorting_score', ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


# Bayesian Average Rating Score

## Sorting Products with 5 Star Rated
## Sorting Products According to Distribution of 5 Star Rating

# puan dağılımlarının üzerinden ağırlıklı bir şekilde olasılıksal ortalama hesaplar
def bayesian_average_rating(n, confidence=0.95):  # n = girilecek olan yıldızların ve bu yıldızlara ait gözlenme frakanslarını ifade etmektedir
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 -(1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

# aralarındaki farkın sebebi düşük puan miktarının diğer kursa göre az olması veya olmaması
df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


# Hybrid Sorting: BAR Score + Diğer Faktörler

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Rating, Comment and Purchase
# - Hybrid Sorting: BAR Score + Diğer Faktörler


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(200)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


# Uygulama: IMDB Movie Scoring & Sorting

df = pd.read_csv("Measurement-Problems/Datasets/movies_metadata.csv", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape


# Vote Average'a Göre Sıralama
df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df['vote_count'] > 400].sort_values("vote_average", ascending=False).head(20)

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[['vote_count']]). \
    transform(df[['vote_count']])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)

df.head()


# IMDB Weighted Rating
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v/(v+M) * r) + (M/(v+M) * C)

df.sort_values("average_count_score", ascending=False).head(20)

# Deadpool filmi için
weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(7.40000, 12.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)


# Inception
weighted_rating(8.1000, 14075.00000, M, C)

# Esaretin bedeli
weighted_rating(8.000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)


# Bayesian Average Rating Score (BAR Score)
df.head()

def bayesian_average_rating(n, confidence=0.95):  # n = girilecek olan yıldızların ve bu yıldızlara ait gözlenme frakanslarını ifade etmektedir
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 -(1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

# Esaretin Bedeli için sırasıyla almış olduğu yıldız sayıları, kaç tane 1 yıldız..., kaç tane  10 yıldız vs
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

df = pd.read_csv("Measurement-Problems/Datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)

