### Rating Products

# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puan Hesaplama
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

# (50+ Saat) Python A-Z: Veri Bilimi Ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv(r"Measurement-Problems/course_reviews.csv")
df.head()
df.shape
df["Rating"].value_counts()
df["Questions Asked"].value_counts()

## Soru soranların sayısına göre verdikleri puan ortalamaları
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                  "Rating": "mean"})

df.head()

# Average

## Ortalama Puan
df["Rating"].mean()


# Time-Based Weighted Average
## Puan Zamanlarına Göre Ağırlıklı Ortalama
df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

current_date = pd.to_datetime("2021-02-10 0:0:0")

# Yapılan yorumları gün cinsinde ifade etmek (3 gün önce vs.)
df["days"] = (current_date - df["Timestamp"]).dt.days

# Son 30 gün içerisinde yapılan yorumların ortalaması
df.loc[df["days"] <= 30, "Rating"].mean()

# 30'dan büyük 90'a küçük eşit olan aralık
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[df["days"] > 180, "Rating"].mean()

#Ağırlık oluşturma
# %28, %26 gibi önem sırasına göre, Not: \ atınca "alt satırdan koda devam edicem" demek
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)
