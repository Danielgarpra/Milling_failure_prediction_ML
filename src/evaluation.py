import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras



with open('../models/model_normal', 'rb') as archivo_entrada:
    model = pickle.load(archivo_entrada)
with open('../models/model_over', 'rb') as archivo_entrada:
    model_over_resampled = pickle.load(archivo_entrada)
with open('../models/model_under', 'rb') as archivo_entrada:
    model_under_resampled = pickle.load(archivo_entrada)
with open('../models/model_normal_pca', 'rb') as archivo_entrada:
    model2 = pickle.load(archivo_entrada)
with open('../models/model_over_pca', 'rb') as archivo_entrada:
    model_over_resampled2 = pickle.load(archivo_entrada)
with open('../models/model_under_pca', 'rb') as archivo_entrada:
    model_under_resampled2 = pickle.load(archivo_entrada)

X_test=pd.read_csv('../data/test/X_test.csv')
y_test=pd.read_csv('../data/test/y_test.csv')

predictions=model.predict(X_test)

#### UNDERSAMPLER:
predictions_under=model_under_resampled.predict(X_test)

#### OVERSAMPLER:
predictions_over=model_over_resampled.predict(X_test)

plt.figure(figsize=(12,2))
plt.subplot(1,3,1)
plt.title('Matrix_confusion_normal')
sns.heatmap(confusion_matrix(y_test,predictions),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(1,3,2)
plt.title('Matrix_confusion_under_resampled')
sns.heatmap(confusion_matrix(y_test,predictions_under),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(1,3,3)
plt.title('Matrix_confusion_over_resampled')
sns.heatmap(confusion_matrix(y_test,predictions_over),cmap='BrBG', annot=True,fmt='.0f',);

#### PCA Models

predictions2=model2.predict(X_test)

predictions_under2=model_under_resampled2.predict(X_test)

predictions_over2=model_over_resampled2.predict(X_test)


### STATISTICS OF THE SIX MODELS:
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.title('Matrix_confusion_normal')
sns.heatmap(confusion_matrix(y_test,predictions),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(2,3,2)
plt.title('Matrix_confusion_under_resampled')
sns.heatmap(confusion_matrix(y_test,predictions_under),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(2,3,3)
plt.title('Matrix_confusion_over_resampled')
sns.heatmap(confusion_matrix(y_test,predictions_over),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(2,3,4)
plt.title('Matrix_confusion_normal PCA')
sns.heatmap(confusion_matrix(y_test,predictions2),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(2,3,5)
plt.title('Matrix_confusion_under_resampled PCA')
sns.heatmap(confusion_matrix(y_test,predictions_under2),cmap='BrBG', annot=True,fmt='.0f',);
plt.subplot(2,3,6)
plt.title('Matrix_confusion_over_resampled PCA')
sns.heatmap(confusion_matrix(y_test,predictions_over2),cmap='BrBG', annot=True,fmt='.0f',);

#And below are the metrics for the two over-resampled models, which have provided the best results:
print(classification_report(y_test, predictions_over))
print(classification_report(y_test, predictions_over2))

results_sk=cross_val_score(model_over_resampled,X_test,y_test,scoring='roc_auc',cv=10)
results_pca=cross_val_score(model_over_resampled2,X_test,y_test,scoring='roc_auc',cv=10)

results_sk.mean(), results_pca.mean()

results_sk=cross_val_score(model_over_resampled,X_test,y_test,scoring='recall',cv=10)
results_pca=cross_val_score(model_over_resampled2,X_test,y_test,scoring='recall',cv=10)

results_sk.mean(), results_pca.mean()

results_sk=cross_val_score(model_over_resampled,X_test,y_test,scoring='precision',cv=10)
results_pca=cross_val_score(model_over_resampled2,X_test,y_test,scoring='precision',cv=10)

results_sk.mean(), results_pca.mean()

#Evaluation ANN
file='../models/model_types.keras'

model=keras.saving.load_model(file)
X_test=pd.read_csv('../data/test/X_test_keras.csv')
y_test=pd.read_csv('../data/test/y_test_keras.csv')

with open('../models/scaler', 'rb') as archivo_entrada:
    scaler = pickle.load(archivo_entrada)

X_test=scaler.transform(X_test)
# Predicitons
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels
y_pred_binary
# Evaluation
print(classification_report(y_test, y_pred_binary))
# Confusion matrix
y_test_lista=np.array(y_test.values)
plt.figure(figsize=(14,3))
title_data=['TWF', 'HDF', 'PWF', 'OSF']

for i in range(y_pred_binary.shape[1]):
  y_pred_binary_clas=list()
  y_test_lista_clas=list()
  for j in range(y_pred_binary.shape[0]):
    if y_pred_binary[j][i]==1:
      y_pred_binary_clas.append(1)
    else:
      y_pred_binary_clas.append(0)
    if y_test_lista[j][i]==1:
      y_test_lista_clas.append(1)
    else:
      y_test_lista_clas.append(0)
  #plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.subplot(1,4,i+1)
  sns.heatmap(confusion_matrix(y_test_lista_clas, y_pred_binary_clas), annot=True, cmap='BrBG', fmt=".2f")
  plt.xticks([])
  plt.title(title_data[i],fontsize=12)
