################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("Machine-Learning/Datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

################################################
# 2. Data Preprocessing & Feature Engineering
################################################
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)  # bağımsız değişkenelrimizi standartlaştırdık

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)


random_user = X.sample(1, random_state=45)  # rastgele bir kullanıcı oluşturuyoruz

knn_model.predict(random_user)  # oluşturulan random user ile diyabet mi değil mi sonucuna bakıyoruz

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred
y_pred = knn_model.predict(X)

# AUC için y_pred
y_prob = knn_model.predict_proba(X)[:, 1]  # 1 sınıfna ait olma olasılıklar

print(classification_report(y, y_pred))  # %83 başarı oranı
# acc 0.83
# f1 0.74

# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model,
                            X, y,  #bağımlı ve bağımsız değişkenler
                            cv=5,  # dördüyle model kur, biriyle test et
                            scoring=["accuracy", "f1", "roc_auc"])  # istediğimiz metrikler

cv_results['test_accuracy'].mean()  # 0.73
cv_results['test_f1'].mean()  # 0.59
cv_results['test_roc_auc'].mean()  # 0.78

# Başarı skorları nasıl arttırılabilir ?
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği (yeni değişkenler türetilebilir)
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

# amaç: şuan olan komşu sayısını değiştirerek, olması gereken en optimumu komşu sayısını bulmak
knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,  # modelimiz
                           knn_params,  # parametre setimiz
                           cv=5,  # hatamızı 5 katlı değerlendiriyoruz
                           n_jobs=-1,  # -1 yapılması durumunda işlemciler en yüksek performans da kullanılır, daha hızlı sonuçlara gider
                           verbose=1).fit(X, y)  # verbose: rapor beklyor musun -> 1(evet) rapor getir

# sonuç: 48 aday geldi, denenecek 48 tane hiperparametre değeri var

knn_gs_best.best_params_  # komşuluk sayısı 17 geldi, bu komşuluk sayısıyla model kurarsam başarısının daha iyi olmasını beklerim

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_)  # iki yıldız(**) kullanarak keyword'ümüzü bu giib durumlarda direkt kullanabiliriz.

cv_results = cross_validate(knn_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.73 -> 0.76
cv_results['test_f1'].mean()  # 0.59 -> 0.61
cv_results['test_roc_auc'].mean()  # 0.78 -> 0.81

random_user = X.sample(1)
knn_final.predict(random_user)
