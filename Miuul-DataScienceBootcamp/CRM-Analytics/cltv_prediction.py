#############################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction #
#############################################

# 1. Data Preparation (Verinin Hazırlanması)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentleri Oluşturulması
# 6. Çalışmanın Fonksiyonlaştırılması


## 1. Veri Hazırlama

## Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numaraç C ile başlıyorsa iptal edilen işlem
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihih ve zamanı
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz Müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):  # Amacı: kendisine girilen değişken için eşik değer belirlemektir
    quartile1 = dataframe[variable].quantile(0.01)  # quantile: çeyreklik hesaplama için
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):  #
    low_limit, upl_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > upl_limit), variable] = upl_limit


df_ = pd.read_excel(r"CRM-Analytics/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()

# Veri Ön İşleme
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]  # iptal edilen ürünleri düşürme
df = df[(df["Quantity"] > 0)]
df = df[(df["Price"] > 0)]

# !!!!!!!!!!!!!!!!!!!!!!!!!
# içerisinde bildirdiğimiz yardımcı fonksiyonu çağıracak, Quantity değişkeni için eşik değerleri hesaplıcak
# daha sonra bu fonksiyonda o eşik değerlerin üzerinde kalan değerleri o eşik değerler ile değiştirecek
# !!!!!!!!!!!!!!!!!!!!!!!!!
replace_with_threshold(df, "Quantity")
replace_with_threshold(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

## Preparation of Lifetime Data Structure (Lifetime Veri Yapısının Hazırlanması)

# recency: Son satın alma ğzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekar eden toplan satın alma sayısı (frequency > 1)
# monetary: satın alma başına ortalama kazanç
# (Not: rfm'dekiler gibi değil)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [
    lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
    # her bir müşterinin, son alışveriş ve ilk alışveriş tarihini birbirinden çıkar ve gün cinsine çevir
    lambda date: (today_date - date.min()).days],  # müşterinin yaşını hesapla
    "Invoice": lambda Invoice: Invoice.nunique(),
    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})  # her bir müşterinin eşsiz kaç tane faturası var (=frequency)

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

# müşteri yaşı değerlerini haftalık cinsine çevirme
cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

## 2. BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# 1 hafta içerisinde en çok satın alma beklediğimiz 10 müşteri kimdir?

# fonksiyona 1 haftalık tahmin yap diyoruz
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

# sklearn'den predict ile de yapılabilir ancak BG-NBD de geçerlidir, Gamma-Gamma da geçerli değildir!!!
bgf.predict(1,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

# her bir müşteri için gerçekleştirilecek ve cltv_df'in içerisine kaydediyoruz
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

# 1 ay içerisinde en çok satın alma beklediğimiz 10 müşteri kimdir?

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])

# 1 aylık periyotta şirketin beklediği satış sayısı
bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?

bgf.predict(4 * 3,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"]).sum()

# Tahmin Sonuçlarının Değerlendirilmesi

plot_period_transactions(bgf)
plt.show()

## 3. GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)


## 4. Calculation of CLTV with BG-NBD and GG Model (BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması)

# Her bir gözlem birimi için Customer Lifetime Value değeri hesaplanacaktır.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)


## 5. Creating the Customer Segment (CLTV'ye Göre Segmentlerin Oluşturulması)

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count", "mean", "sum"})


## 6. Functionalization (Çalışmanın Fonksiyonlatırılması)

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")

