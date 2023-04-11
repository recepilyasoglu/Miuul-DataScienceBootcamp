###########################################
############# RECEP İLYASOĞLU #############
###########################################

############ Classification Model Evaluation ############

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib

matplotlib.use("Qt5Agg")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', None)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn import metrics

## Görev 1: Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma modeli oluşturulmuştur.
# 10 test verisi gözleminin gerçek değerleri ve modelin tahmin ettiği olasılık değerleri verilmiştir.

# - Eşik değerini 0.5 alarak confusion matrix oluşturunuz.

# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız

df = pd.DataFrame(data={"Gerçek Değer": (1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                        "Model Olasılık Tahmini": (0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25)})

df

df["Tahmin Edilen Değer"] = [1 if col > 0.50 else 0 for col in df["Model Olasılık Tahmini"]]
df

y = df["Gerçek Değer"]
X = df["Tahmin Edilen Değer"]

confusion_matrix = metrics.confusion_matrix(y, X)
accuracy_score(y, X)
print(classification_report(y, X))
# accuracy = 0.70
# precision = 0.80
# recall = 0.67
# f1 score = 0.73


# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
# cm_display.plot()
# plt.show()


## Görev 2: Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur. %90.5 doğruluk
# oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır. Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi
# olmamış, iş birimi modelin başarısız olduğunu iletmiştir. Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir.

# Buna göre;
# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.
# - Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız


Accuracy = (5+900) / 1000
Precision = 5 / (5+90)
Recall = 5 / (5+5)
F1_Score = 2 * (0.05 * 0.50) / (0.05 + 0.50)

print("Accuracy..: ", Accuracy, "\n"
      "Recall..:", Recall, "\n"
      "Precision..:", Precision, "\n"
      "F1_Score..:", F1_Score)


# - Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız
# Veri Bilimi ekibinin gözünden kaçırdığı durum verilerin dengesiz olması durumu.
