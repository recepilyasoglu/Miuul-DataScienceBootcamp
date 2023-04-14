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





