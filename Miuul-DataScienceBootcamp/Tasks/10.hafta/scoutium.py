#####################################################
################## RECEP İLYASOĞLU ##################
#####################################################

# Scouting Classification with Machine Learning


## Business Problem: According to the scores given to the characteristics of the football players watched by the Scouts, which class of the players
# (average, highlighted) player guessing.
# (Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.)


## Dataset Story
# According to the characteristics of the football players observed in the matches from the data set Scoutium, the football players evaluated by the scouts,
# It consists of information about the features scored and their scores.
# (Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
# içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.)


import joblib
import warnings
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

## GÖREVLER

# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

attributes = pd.read_csv("Miuul-DataScienceBootcamp/Tasks/10.hafta/scoutium_attributes.csv", sep=";")
labels = pd.read_csv("Miuul-DataScienceBootcamp/Tasks/10.hafta/scoutium_potential_labels.csv", sep=";")

attributes.head()
labels.head()

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

data = pd.merge(attributes, labels, on=["task_response_id", 'match_id', "evaluator_id", "player_id"])
data.head()

df = data.copy()
df.head()
df.shape
df.describe().T
df.isnull().sum()

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[(df["position_id"] != 1)]

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df = df[(df["potential_label"] != "below_average")]
df["potential_label"].value_counts()

df.groupby("potential_label").mean()

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.

## Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
## “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

pivot_scoutium = df.pivot_table(index=["player_id", "position_id", "potential_label"],
                                columns=["attribute_id"],
                                values="attribute_value")

pivot_scoutium.head()
pivot_scoutium.columns.name

## Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

pivot_scoutium = pivot_scoutium.reset_index()
pivot_scoutium.columns = pivot_scoutium.columns.map(str)
pivot_scoutium.dtypes

# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
binary_cols = [col for col in pivot_scoutium.columns if pivot_scoutium[col].dtype not in ["int64", "float64"]
               and pivot_scoutium[col].nunique() == 2]

pivot_scoutium[binary_cols].value_counts()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(pivot_scoutium, col)

pivot_scoutium[binary_cols].value_counts()

# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
pivot_scoutium.dtypes
pivot_scoutium["potential_label"] = pivot_scoutium["potential_label"].astype("int64")

num_cols = [col for col in pivot_scoutium.columns if pivot_scoutium[col].dtypes in ["int64", "float64"]]

num_cols = [col for col in num_cols if col != "potential_label"]

num_cols = num_cols[2:]

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

scaler = StandardScaler()

pivot_scoutium[num_cols] = scaler.fit_transform(pivot_scoutium[num_cols])

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = pivot_scoutium["potential_label"]
X = pivot_scoutium.drop(["potential_label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

total_precision = base_models(X, y, scoring="precision")
total_roc_auc = base_models(X, y, scoring="roc_auc")
total_f1 = base_models(X, y, scoring="f1")
total_recall = base_models(X, y, scoring="recall")
total_accuracy = base_models(X, y, scoring="accuracy")

results = pd.concat([total_precision, total_roc_auc, total_f1, total_recall, total_accuracy], axis=1)

# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
plot_importance(dt_model, X_train)
# plot_importance(log_model, X_train)
