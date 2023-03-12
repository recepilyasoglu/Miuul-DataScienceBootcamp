### Sorting Products

# Uygulama: Kurs Sıralama
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("Measurement-Problems/product_sorting.csv")
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

