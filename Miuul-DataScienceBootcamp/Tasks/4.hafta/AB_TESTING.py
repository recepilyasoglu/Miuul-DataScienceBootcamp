#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
import itertools
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.width", 500)

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
test_group = pd.read_excel("Tasks/4.hafta/ab_testing.xlsx", sheet_name="Test Group")
control_group = pd.read_excel("Tasks/4.hafta/ab_testing.xlsx", sheet_name="Control Group")

test_group.head()
test_group.describe().T
test_group.dtypes
test_group.shape

control_group.head()
control_group.describe().T
control_group.dtypes
control_group.shape

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.


df = pd.concat([test_group, control_group], axis=0).reset_index()
df.head()

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0: M1 = M2
# H1: M1 != M2

# 2. Varsayım Kontrolü

# Normallik Varsayımı
# Varyans Homojenliği


## Normallik Varsayımı

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: ...sağlanmamaktadır.


## Varyans Homjenliği Varsayımı

# H0: Varyanslar Homojendir.
# H1: Varyanslar Homojen Değildir..

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
test_group["Purchase"].mean()
control_group["Purchase"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik Varsayımı
control_stat, p_control = shapiro(control_group['Purchase'])
test_stat, p_test = shapiro(test_group['Purchase'])
p_val = 0.05

if p_control < p_val:
    print('Kontrol grubu normal dağılmamıştır. (p-value={})'.format(p_control))
else:
    print('Kontrol grubu normal dağılmıştır. (p-value={})'.format(p_control))


if p_test < p_val:
    print('Test grubu normal dağılmamıştır. (p-value={})'.format(p_test))
else:
    print('Test grubu normal dağılmıştır. (p-value={})'.format(p_test))

# Varyans Homojenliği
stat_levene, p_levene = levene(control_group['Purchase'], test_group['Purchase'])

if p_levene < p_val:
    print(f'İki grubun varyansı eşit değildir. (p-value={p_levene})')
else:
    print(f'İki grubun varyansı eşittir. (p-value={p_levene})')

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

stat, p = ttest_ind(control_group['Purchase'], test_group['Purchase'], equal_var=True)

if p < p_val:
    print('İki grup arasında anlamlı bir fark vardır. (p-value={})'.format(p))
else:
    print('İki grup arasında anlamlı bir fark yoktur. (p-value={})'.format(p))


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Normallik varsayımı için shapiro, Varyans Homojenliği Varsayımı için levene,
# Son olarak da elde edilen sonuca göre T-Testi kullanıldı.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# -> İki grubun ortalamaları arasında anlamlı bir fark yoktur.
