import streamlit as st
import pickle
import numpy as np
from tensorflow import keras

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
model_type=keras.models.load_model('./models/model_types.keras')
scaler=load_model('./models/scaler.pkl')

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
        features = np.array([[input_1, input_2, input_3, input_4, input_5, input_6_H, input_6_M,input_6_L]])
        prediction = model.predict(features)[0]
        predict_proba=model.predict_proba(features)[0]

        st.subheader("📢 Resultado de la Predicción")
        if prediction == 0:
            st.success("✅ La fresadora **funcionará correctamente**.")
            st.success(f"Con un {predict_proba[0] * 100:.0f}% de probabilidad")        
        else:
            st.error("⚠️ **¡Cuidado!** Se prevé una **falla** en la fresadora.")
            st.error(f"Con un {predict_proba[1] * 100:.0f}% de probabilidad")

            # Estudiamos el tipo de fallo que se va a producir:
            
            features2 = np.array([[input_1, input_2, input_3, input_4, input_5, input_6_H, input_6_M,input_6_L]])
            prediction2 = model_type.predict(scaler.transform(features2))[0]
            if prediction2[0]>0.5:
                st.error('Se va a producir fallo por desgaste')
                st.error(f"Con un {prediction2[0] * 100:.0f}% de probabilidad")        
                
                st.success("Cambia la herramienta")
                #Aquí habrá que 'reiniciar' el proceso (tool wear)             
            if prediction2[1]>0.5:
                st.error('Se va a producir fallo por disipación de calor')
                st.error(f"Con un {prediction2[1] * 100:.0f}% de probabilidad")        
                st.success("Haz algo para la temperatura")

            if prediction2[2]>0.5:
                st.error('Se va a producir fallo por potencia')
                st.error(f"Con un {prediction2[2] * 100:.0f}% de probabilidad")        

                st.success("Cambia la velocidad rotacional")
            if prediction2[3]>0.5:
                st.error('Se va a producir fallo por sobreesfuerzo')
                st.error(f"Con un {prediction2[3] * 100:.0f}% de probabilidad")        

                st.success("Baja el par torsión o cambia la herramienta")                

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")





# Footer
st.markdown("---")
st.markdown("🔍 **Futuras implementaciones:** Predicción del tipo de fallo (TWF, HDF, PWF, OSF).")
