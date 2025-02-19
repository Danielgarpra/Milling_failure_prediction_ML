import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split ,KFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow import keras

df=pd.read_csv('../data/processed/data_processed.csv')
## MODELING
### ***FIRST CASE: STUDY OF MACHINE FAILURE PREDICTION:***
X=df.drop(['Product ID','Machine failure','TWF','HDF','PWF','OSF','Power [W]','Disipation'],axis=1)
y=df['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv('./data/train/X_train.csv')
y_train.to_csv('./data/train/y_train.csv')
X_test.to_csv('./data/test/X_test.csv')
y_test.to_csv('./data/test/y_test.csv')

oversampler = RandomOverSampler(random_state=42)
X_over_resampled, y_over_resampled = oversampler.fit_resample(X_train, y_train)

X_over_resampled.to_csv('./data/train/X_train_over.csv')
y_over_resampled.to_csv('./data/train/y_train_over.csv')
undersampler = RandomUnderSampler(random_state=42)
X_under_resampled, y_under_resampled = undersampler.fit_resample(X_train, y_train)

X_under_resampled.to_csv('./data/train/X_train_under.csv')
y_under_resampled.to_csv('./data/train/y_train_under.csv')

pipe = Pipeline(steps=[("selectkbest", SelectKBest()),
    ("scaler", StandardScaler()),
    ('classifier', svm.SVC())
])

random_forest_params = {
    'selectkbest__k': [3,4,5,'all'],
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'classifier': [RandomForestClassifier(random_state=100)],
    'classifier__max_depth': [2,3,4]
}

svm_param = {
    'selectkbest__k': [3,4,5,'all'],
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'classifier': [svm.SVC(probability=True, random_state=42)],
    'classifier__C': [ 1, 5, 10, 100],
}


search_space = [
    random_forest_params,
    svm_param,
]

clf = GridSearchCV(estimator = pipe,
                  param_grid = search_space,
                  cv = KFold(10),
                  scoring='recall',
                  verbose=2,
                  n_jobs=-1)

clf.fit(X_train, y_train)
model=clf.best_estimator_

predictions=model.predict(X_test)
predictions_proba_normal = model.predict_proba(X_test)

#### UNDERSAMPLER:
clf.fit(X_under_resampled, y_under_resampled)
model_under_resampled=clf.best_estimator_

predictions_under=model_under_resampled.predict(X_test)
predictions_proba_under = model_under_resampled.predict_proba(X_test)

#### OVERSAMPLER:
clf.fit(X_over_resampled, y_over_resampled)
model_over_resampled=clf.best_estimator_

predictions_over=model_over_resampled.predict(X_test)
predictions_proba_over = model_over_resampled.predict_proba(X_test)


#### PCA Models
pipe2 = Pipeline(steps=[("pca", PCA()),
    ("scaler", StandardScaler()),
    ('classifier', svm.SVC())
])

random_forest_params2 = {
    'pca__n_components': [3,4,5,None],
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'classifier': [RandomForestClassifier(random_state=100)],
    'classifier__max_depth': [2,3,4]
}

svm_param2 = {
    'pca__n_components': [3,4,5,None],
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'classifier': [svm.SVC(probability=True, random_state=42)],
    'classifier__C': [ 1, 5, 10, 100]
}

search_space2 = [
    random_forest_params2,
    svm_param2
]

clf2 = GridSearchCV(estimator = pipe2,
                  param_grid = search_space2,
                  cv = KFold(10),
                  scoring='recall',
                  verbose=2,
                  n_jobs=-1)

clf2.fit(X_train, y_train)
model2=clf2.best_estimator_
model2
predictions2=model2.predict(X_test)

clf2.fit(X_under_resampled, y_under_resampled)
model_under_resampled2=clf2.best_estimator_
model_under_resampled2
predictions_under2=model_under_resampled.predict(X_test)

clf2.fit(X_over_resampled, y_over_resampled)
model_over_resampled2=clf2.best_estimator_
model_over_resampled2
predictions_over2=model_over_resampled2.predict(X_test)


#### Save the models:

filename = './models/model_normal.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model, archivo_salida)

filename = './models/model_under.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model_under_resampled, archivo_salida)

filename = './models/model_over.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model_over_resampled, archivo_salida)

filename = './models/model_normal_pca.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model2, archivo_salida)

filename = './models/model_under_pca.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model_under_resampled2, archivo_salida)

filename = './models/model_over_pca.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(model_over_resampled2, archivo_salida)
### ***SECOND CASE: STUDY OF FAILURE TYPE PREDICTION:***

df_type=df[df['Machine failure']==1]
df_type.to_csv('./data/processed/processed_data_type.csv')

# Define input and output variables
X_t = df_type.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'Machine failure','Disipation','Power [W]', 'Product ID'])  # Features
y_t = df_type[['TWF', 'HDF', 'PWF', 'OSF']]  # labels
X_t.head(2)
y_t.head(2)

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
X_train.to_csv('./data/train/X_train_keras.csv')
y_train.to_csv('./data/train/y_train_keras.csv')
X_test.to_csv('./data/test/X_test_keras.csv')
y_test.to_csv('./data/test/y_test_keras.csv')

# Normalize data
scaler = StandardScaler()
X_t_scaled = scaler.fit_transform(X_t)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_t_scaled, y_t, test_size=0.2, random_state=42)

# Build the model in Keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='sigmoid')  # 4 outputs for multi-label classification
])
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],

              )
earlystopping = keras.callbacks.EarlyStopping(patience=5)
# Train the model
history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2 ,
                    callbacks = [earlystopping]
)
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
# Predicitons
y_pred = model.predict(X_test)
np.round(y_pred,2)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels
y_pred_binary


#Saving the model:
model.save('model_types.keras')
import pickle

filename = 'scaler.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)