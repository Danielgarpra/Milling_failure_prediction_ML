import streamlit as st
import pickle
import numpy as np

# Cargar el modelo preentrenado
@st.cache_resource  # Cachear el modelo para mejorar el rendimiento
def load_model():
    try:
        with open("./models/model_over.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# Título de la app
st.title("🔧 Predicción del Correcto Funcionamiento de tu Fresadora")
st.markdown("### 🏭 Introduce los datos del modelo")

# Crear formulario para entrada de datos
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        input_1 = st.number_input("🌡️ Air Temperature", value=277.0, min_value=0.0)
        input_2 = st.number_input("🔥 Process Temperature", value=300.0, min_value=0.0)
        input_3 = st.number_input("⚙️ Rotational Speed", value=1000.0, step=10.0, min_value=0.0)

    with col2:
        input_4 = st.number_input("🔩 Torque", value=40.0, step=1.0, min_value=0.0)
        input_5 = st.number_input("🔧 Tool Wear", value=0.0, step=1.0, min_value=0.0)
        input_6 = st.selectbox("🛠️ Tipo de Herramienta", options=["L", "M", "H"], index=0)

    # Botón de envío dentro del formulario
    submit = st.form_submit_button("📊 Hacer Predicción")

# Mapeo del tipo de herramienta
tool_mapping = {"L": [1, 0, 0], "M": [0, 1, 0], "H": [0, 0, 1]}
input_6_L, input_6_M, input_6_H = tool_mapping[input_6]

# Ejecutar predicción cuando el usuario envíe el formulario
if submit:
    try:
        features = np.array([[input_1, input_2, input_3, input_4, input_5, input_6_L, input_6_M, input_6_H]])
        prediction = model.predict(features)[0]

        st.subheader("📢 Resultado de la Predicción")
        if prediction == 0:
            st.success("✅ La fresadora **funcionará correctamente**.")
        else:
            st.error("⚠️ **¡Cuidado!** Se prevé una **falla** en la fresadora.")
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

# Footer
st.markdown("---")
st.markdown("🔍 **Futuras implementaciones:** Predicción del tipo de fallo (TWF, HDF, PWF, OSF).")