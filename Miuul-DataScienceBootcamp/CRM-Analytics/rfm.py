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
