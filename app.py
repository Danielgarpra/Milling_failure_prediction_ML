import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cargar el modelo preentrenado
with open("model_final.pkl", "rb") as file:
    model = pickle.load(file)

# Título de la app
st.title("Predicción con Machine Learning")

# Crear formulario para entrada de datos
st.header("Introduce los datos del modelo")
input_1 = st.number_input("Valor de la air temperature característica", value=277.0)
input_2 = st.number_input("Valor de la process temperature característica", value=277.0)
input_3 = st.number_input("Valor de la rotational speed característica", value=1000.0)
input_4 = st.number_input("Valor de la torque característica", value=0.0)
input_5 = st.number_input("Valor de la tool wear característica", value=0.0)


# Botón para hacer predicción
if st.button("Hacer Predicción"):
    # Crear un DataFrame con las características
    features = np.array([[input_1, input_2,input_3,input_4,input_5]])
    prediction = model.predict(features)
    
    st.success(f"La predicción del modelo es: {prediction[0]}")