################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model

import warnings
import matplotlib
matplotlib.use("Qt5Agg")
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Exploratory Data Analysis
################################################

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

################################################
# 3. Modeling using CART
################################################

df = pd.read_csv("Machine-Learning/Datasets/diabetes.csv")
df.head()

y = df["Outcome"]  # bağımlı değişken
X = df.drop(["Outcome"], axis=1)  # bağımsız değişken

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion Matrix için y_pred
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))  # 1 geldi ama bu pek mümkün değil

# AUC
roc_auc_score(y, y_prob)

# başarımı nasıl daha dopru değerlendirebilirim?

###########################################
# Holdout Yöntemi ile Başarı Değerlendirme
###########################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))  # yine 1 geldi
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = cart_model.predict(X_test)  # yukarda train üzerinde kurduğumuz modele daha önce hiç görmediğimiz test setini verdik
y_prob = cart_model.predict_proba(X_test)[:, 1]  # sadece 1’lerin olasılığını aldık
print(classification_report(y_test, y_pred))  # şimdi daha iyi geldi
roc_auc_score(y_test, y_prob)  # 0.65

# sonuç: model göremediği veri de berbat -> Overfitting


##########################################
# CV ile Başarı Değerlendirme
##########################################
# cross validation ile yapıcaz şimdi çünkü sıkıntı çıktı
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)  # ek fit de demeyebilirdik cross validate zaten görmüyoronu, kendi oluşturuyor çünkü aşağıda

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347


################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()
# CART ile ilgili ilgilendiğimiz hiper parametreler; max_depth ve min_samples_split

cart_params = {"max_depth": range(1, 11),  # bu değeleri girerken ön tanımlı değerlere bakıyoruz,
               "min_samples_split": range(2, 20)}  # -> ön tanımlı değerlerin etrafındaki değerleri gir

## en iyi parametreleri vermesi için kullanıyoruz
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
# bu iki hiperparametrenin olası 180 tane kombinasyonu varmış

cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1, random_state=45)  # verimizden rastgele kullanıcı çektik

cart_best_grid.predict(random)  # verilen rastgele kullanıcı için diyabet olumlu (1) dedi

# gGridSearchCV ile de final model'siz predict yapılabilir,
# ama farkındalık açısından final model kurulması daha iyi


################################################
# 5. Final Model
################################################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

# bir diğer yolu, parametreleri var olan modele set_params diyerek set ediyoruz
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

# final modelin cv hatası
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


################################################
# 6. Feature Importance
################################################

cart_final.feature_importances_  # değişkenlerin önem düzeyleri geldi

def plot_importance(model, features, num=len(X), save=False):
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

# num değeri = görselleştirmel istediğimiz değişken sayısı
# cart_final = oluşturulan modelimiz
# X = feature'larımız

plot_importance(cart_final, X, num=5)


#############################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
#############################################################

# seçilen parametreye göre öğrenme eğrilerini yazdırma
train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),  # 1ile 11 arasındaki derinlikleri denesin bana raporlasın
                                           scoring="roc_auc",  # roc_auc metriği açısından
                                           cv=10)  # 5 de denilebilir

# değişkenlerin max_depth de yer alan her bir değer için 10 array geldi
# array'in içerisinde ise 10 katlı cross validation'ın sonuçları

mean_train_score = np.mean(train_score, axis=1)  # train_score için ortalamalarını aldık
mean_test_score = np.mean(test_score, axis=1)  # test_score için ortalamalarını aldık


# train hatası ve test hatası birlikte görselleştirilir
# ve ayrım noktasından itibaren karar verilir.


plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')  # test de denilebilir burada

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

# buraya fikir edinebilmek için bakıyoruz,
# zaten hiperparametre optimizasyon kısmında optimum değerlerimizi bulmuştuk


# Fonksiyonlaştırılması
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

# birden fazla parametreler için, liste oluştuyoruz
cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

# listenin elemanlarında gezip validation curve uyguluyoruz
for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])


################################################
# 8. Visualizing the Decision Tree
################################################

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()  # 5 tane dallanma işlemi olmuş, 5 seviye gerçekleşmiş yani


################################################
# 9. Extracting Decision Rules
################################################

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)  # yukarıda çıkardığımız karar kuralları geldi


################################################
# 10. Extracting Python Codes of Decision Rules
################################################

# canlı ortamlarda kullanabilmemiz adına

print(skompile(cart_final.predict).to('python/code'))
# python dilinde karar kurallarımız

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))
# sql dilinde karar kurallarımız (en temizi)

print(skompile(cart_final.predict).to('excel'))
# excel kodları üzerinden karar kurallarımız


################################################
# 11. Prediction using Python Codes
################################################

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]  # gelen hasta bilgilerinin buynlar olduğunu varsayıyoruz

predict_with_rules(x)  # tahmnilememiz sonucu "diyabet değildir" sonucu aldık

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)  # tahmnilememiz sonucu "diyabettir" sonucu aldık


################################################
# 12. Saving and Loading Model
################################################

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("Machine-Learning/cart_final.pkl")  # diskten yükle

x = [12, 13, 20, 23, 4, 55, 12, 7]  # yeni gelen model için veriler

cart_model_from_disc.predict(pd.DataFrame(x).T)  # x'in Transpozunu alarak dataframe'e çevirip okuttuk
# sonuç: veriler bu gözlem birimi(x) diyabettir (1)

