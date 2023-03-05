##################################
# Customer Segmentation with RFM #
##################################

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Functionalization of the whole process


## 1. Business Problem
# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

## Dataset
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numaraç C ile başlıyorsa iptal edilen işlem
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihih ve zamanı
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz Müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke


## 2. Data Understanding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

data = pd.read_excel(r"C:\Users\reco1\PycharmProjects\Miuul-DataScienceBootcamp\CRM-Analytics\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = data.copy()
df.head()
df.shape
df.isnull().sum()

# eşsiz ürün  sayısı nedir?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head()

# her bir üründen toplam ne kadar sipariş verildiği
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# eşsiz fatura sayısı
df["Invoice"].nunique()

# her bir ürünün tam fiyatı
# kaç tane satıldığı * fiyatı
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Invoice başına toplan ne kadar ödendiği
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()


## 3. Data Preparation

df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.describe().T

# başında C olmayanlar gelsin çünkü C olanlar, iptal edilen ürünler ve negatif değerli ürünler
df = df[~df["Invoice"].str.contains("C", na=False)]


## 4. Calculating RFM Metrics

# Recency, Frequency, Monetary
df.head()
df["InvoiceDate"].max()

# analizi yaptığımız gün olarak belirliyoruz ki
# müşterini recency değerine bakabilelim
today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     "Invoice": lambda Invoice: Invoice.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
rfm.head()

rfm.columns = ["recency", "frequency", "monetary"]

rfm.describe().T

# monetary değeri 0 dan büyük olanları seç
rfm = rfm[rfm["monetary"] > 0]
rfm.shape


## 5. Calculating RFM Scores

# recency değerini küçükten büyüğe sırala, 5 parçaya böl
# bunu tersten sıralıyoruz yani 5 olan müşteri sıklık açısından 1'e göre daha kötü,
# daha az sıklıkla alışveriş yapmıştır yani
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# burda ve frequency'de normal sıralama yapıyoruz, değeri 5 olan en çok para bırakan müşteridir gibi
# yani küçük gördüğüne küçük puan, büyük gördüğüne büyük puan ver
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# belirtilen aralıklarda kendini tekrar eden birden fazla değer var
# bunun için rank methodunun kullanıyoruz, ilk gördüğünü ilk sınıfa ata
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

rfm.describe().T

# şampiyon müşteriler
rfm[rfm["RFM_SCORE"] == "55"]

# görece önemi daha düşük müşteriler
rfm[rfm["RFM_SCORE"] == "11"]


## 6. Creating & Analysing RFM Segments
# regex

# RFM Naming
seg_map = {
    r"[1-2][1-2]": "hibernating",  # birinci elemanında 1 yada 2, ikinci elemanında 1 yada 2 görürsen o isimlendirmeyi yap
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "need_attention",
    r"33": "about_to_sleep",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",  # birinci elemanında 4, ikinci elemanında 1 görürsen o isimlendirmeyi yap
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}

# RFM_SCORE da değişkenleri seg_map deki değerler ile değiştir ve onlara da dağılım uygula yukarıdaki gibi
rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

# segmentlerin analizi
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "need_attention"].head()
rfm[rfm["segment"] == "cant_loose"].head()

#Id bilgileri index ile geldi
rfm[rfm["segment"] == "new_customers"].index

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# ondalıklardan kurtulduk
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

new_df.to_csv("new_customers.csv")

# tüm segment bilgileri gider
rfm.to_csv("rfm.csv")


## 7. Functionalization of the whole process

def create_rfm(dataframe, csv=False):

    # VERININ HAZIRLANMASI
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]


    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2010, 12, 11)
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                         "Invoice": lambda Invoice: Invoice.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]


    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                        rfm["frequency_score"].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r"[1-2][1-2]": "hibernating",
        # birinci elemanında 1 yada 2, ikinci elemanında 1 yada 2 görürsen o isimlendirmeyi yap
        r"[1-2][3-4]": "at_Risk",
        r"[1-2]5": "cant_loose",
        r"3[1-2]": "need_attention",
        r"33": "about_to_sleep",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",  # birinci elemanında 4, ikinci elemanında 1 görürsen o isimlendirmeyi yap
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions",
    }

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:     # eğer csv argümanında true yazıyorsa csv dosyası oluştur
        rfm.to_csv("rfm.csv")

    return rfm

df = data.copy()

rfm_new = create_rfm(df, csv=True)

