import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
import plotly.graph_objects as go

# Configuración de estilo
st.set_page_config(page_title="Panel de Control Industrial", layout="wide", page_icon="🏭")

# Cargar el modelo preentrenado
@st.cache_resource  # Cachear el modelo para mejorar el rendimiento
def load_model(dir):
    try:
        with open(dir, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model('./models/model_over.pkl')
model_type = keras.models.load_model('./models/model_types.keras')
scaler = load_model('./models/scaler.pkl')

# Título de la app
st.markdown("""
    <h1 style='text-align: center; color: #FFA500; font-family: Arial, sans-serif;'>
        🏭 Panel de Control - Predicción de Fallos en Fresadora
    </h1>
""", unsafe_allow_html=True)

# Crear layout de columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Parámetros de la Máquina")
    input_1 = st.slider("🌡️ Air Temperature (K)", 277.0, 330.0, 300.0, 0.1)
    input_2 = st.slider("🔥 Process Temperature (K)", 277.0, 340.0, 310.0, 0.1)
    input_3 = st.slider("⚙️ Rotational Speed (rpm)", 0.0, 5000.0, 1000.0, 1.0)
    input_4 = st.slider("🔩 Torque (Nm)", 0.0, 100.0, 40.0, 1.0)
    input_5 = st.slider("🔧 Tool Wear (min)", 0.0, 500.0, 0.0, 1.0)
    input_6 = st.radio("🛠️ Tipo de Herramienta", ["L", "M", "H"], horizontal=True)
    submit = st.button("📊 Hacer Predicción", use_container_width=True)

with col2:
    st.markdown("### 📊 Indicador de Estado")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=input_2,
        title={"text": "Temperatura del Proceso (K)"},
        gauge={"axis": {"range": [280, 350]}, "bar": {"color": "red"}}
    ))
    st.plotly_chart(fig)

# Mapeo del tipo de herramienta
tool_mapping = {"L": [1, 0, 0], "M": [0, 1, 0], "H": [0, 0, 1]}
input_6_L, input_6_M, input_6_H = tool_mapping[input_6]

# Ejecutar predicción cuando el usuario envíe el formulario
if submit:
    try:
        features = np.array([[input_1, input_2, input_3, input_4, input_5, input_6_H, input_6_M, input_6_L]])
        prediction = model.predict(features)[0]
        predict_proba = model.predict_proba(features)[0]

        st.markdown("### 📢 Resultado de la Predicción")
        if prediction == 0:
            st.success("✅ La fresadora **funcionará correctamente**.")
            st.success(f"Con un {predict_proba[0] * 100:.0f}% de probabilidad")
        else:
            st.error("⚠️ **¡Cuidado!** Se prevé una **falla** en la fresadora.")
            st.error(f"Con un {predict_proba[1] * 100:.0f}% de probabilidad")

            features2 = np.array([[input_1, input_2, input_3, input_4, input_5, input_6_H, input_6_M, input_6_L]])
            prediction2 = model_type.predict(scaler.transform(features2))[0]
            
            if prediction2[0] > 0.5:
                st.error('Se va a producir fallo por desgaste')
                st.success("Cambia la herramienta")
            if prediction2[1] > 0.5:
                st.error('Se va a producir fallo por disipación de calor')
                st.success("Haz algo para la temperatura")
            if prediction2[2] > 0.5:
                st.error('Se va a producir fallo por potencia')
                st.success("Cambia la velocidad rotacional")
            if prediction2[3] > 0.5:
                st.error('Se va a producir fallo por sobreesfuerzo')
                st.success("Baja el par torsión o cambia la herramienta")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")