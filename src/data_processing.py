import pandas as pd
import numpy as np

## DATA LOAD
#Datasets: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
df=pd.read_csv('./data/raw/ai4i2020.csv',index_col=0)

## FEATURE ENGINEERING
df=df.drop(columns='RNF')
df['Power [W]']=df['Torque [Nm]']*(df['Rotational speed [rpm]']*(2*np.pi/60))
df['Disipation']=df['Process temperature [K]']-df['Air temperature [K]']
df = pd.get_dummies(df, columns=['Type'])
df_aux=df[['Machine failure','TWF','HDF','PWF','OSF']]
df=df.drop(columns=['Machine failure','TWF','HDF','PWF','OSF'])
df['Machine failure']=df_aux['Machine failure']
df['TWF']=df_aux['TWF']
df['HDF']=df_aux['HDF']
df['PWF']=df_aux['PWF']
df['OSF']=df_aux['OSF']

df.to_csv('./data/processed/data_processed.csv')