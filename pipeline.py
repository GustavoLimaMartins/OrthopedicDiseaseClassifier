import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml as fo
# Integrando a API do openml com o dataset
db = fo(data_id=1523)
# Convertendo o conjunto de dados em dataframe (apenas atributos preditivos)
df = pd.DataFrame(data=db['data'])
# Colocando o atributo-alvo no dataframe com o nome da doença
labels = {'1':'Disk Hernia', '2':'Normal', '3':'Spondylolisthesis'}
df_labels = list()
for i in range(len(df.index)): df_labels.append(labels[str(db.target[i])])
df['diagnostic'] = df_labels
# Consultando a dimensionalidade dos dados
print('Dimensão:', df.shape)
# Topologia dos dados
print('Topologia:\n', df.info())
# Média dos dados por classe
print('Estatísicas:\n', df.groupby('diagnostic').mean())
# Comportamento de pares ordenados entre as variáveis no plano cartesiano
import seaborn as sns
sns.pairplot(df, hue='diagnostic')
plt.show()
# Maiores correlações existentes entre atributos preditivos
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f')
plt.show()
bigtest_corr = (('V1', 'V4'), ('V1', 'V3'), ('V1', 'V6'), ('V1', 'V2'))
# 0.81, 0.72, 0.64, 0.63
# Histograma dos dados
df.hist()
plt.show()
# Padronização da escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns='diagnostic')))
sns.boxplot(df_scaled)
plt.show()
# Remoção de outliers coletiva
indices = []
for i in range(len(df_scaled.columns)):
    limit_inf = df_scaled.loc[df_scaled[i] < -2.5].index
    limit_sup = df_scaled.loc[df_scaled[i] > 3].index
    df_scaled.drop(limit_inf, inplace=True)
    df_scaled.drop(limit_sup, inplace=True)
    indices.extend(limit_inf)
    indices.extend(limit_sup)
sns.boxplot(df_scaled)
plt.show()

from sklearn.model_selection import train_test_split
df.drop(labels=indices, inplace=True, axis=0)
X = df.drop(columns=['diagnostic'])
y = df['diagnostic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Técnica de oversampling (maximização de instâncias das classes minoritárias)
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train_os, y_train_os = oversample.fit_resample(X_train_scaled, y_train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_os, y_train_os)
y_model = model.predict(X_test_scaled)
print(y_train_os.value_counts())
print(classification_report(y_true=y_test, y_pred=y_model))

matriz_confusao = confusion_matrix(y_true = y_test,
                                   y_pred = y_model,
                                   labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])

# Plotando uma figura com a matriz de confusão
figure = plt.figure(figsize=(15, 5))
disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao,
                              display_labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
disp.plot(values_format='d')
plt.show()

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np


error = [] #armazenar os erros
# Calculating error for K values between 1 and 15
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_os, y_train_os)
    pred_i = knn.predict(X_test_scaled)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate n Value')
plt.xlabel('n Value')
plt.ylabel('Mean Error')
plt.show()

param_grid = {'n_neighbors': [3,4,8,10,12,14],
              'weights': ['distance'],
              'metric': ['euclidean', 'manhattan']
}

gs_metric = make_scorer(accuracy_score, greater_is_better=True)
try:
    # Configuração de KFold.
    kfold  = KFold(n_splits=5, shuffle=True) 
    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid=param_grid,
                        scoring=gs_metric,
                        cv=kfold, n_jobs=5, verbose=3
    )
    
    grid.fit(X_train_os, y_train_os)
    print('KNN', grid.best_params_)
    best_p = grid.best_params_
except:
    print('stopped')

# KNN
knn = KNeighborsClassifier(n_neighbors=best_p['n_neighbors'], metric=best_p['metric'], weights=best_p['weights'])
knn.fit(X_train_os, y_train_os)
y_knn = knn.predict(X_test_scaled)
print(accuracy_score(y_test, y_knn))
