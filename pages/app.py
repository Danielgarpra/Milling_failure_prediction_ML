import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
import plotly.graph_objects as go
from  streamlit_vertical_slider import vertical_slider 
from time import sleep
from stqdm import stqdm


# Configuración de estilo
st.set_page_config(page_title="Milling machine performance prediction", layout="wide", page_icon="🏭",initial_sidebar_state="expanded", menu_items={

        'About': "https://github.com/Danielgarpra/Milling_failure_prediction_ML"
    })

# Definir las métricas y simulaciones
features=["🌡️ Air temperature (C)","🔥 Process temperature (C)",'⚙️ Rotational Speed (rpm)',"🔩 Torque (Nm)","⌛ Tool Wear (min)","🛠️ Type of tool",]
metric = ['ºC', 'ºC', 'rpm', 'Nm', 'min', 'type']
sim1 = [27, 38, 2000, 40, 160, 'L']
sim2 = [33, 40, 1000, 20, 230, 'M']
sim3 = [25, 36, 3000, 60, 40, 'H']

# Crear las posibles simulaciones
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    simulacion_1 = st.button("SIMULATION 1", use_container_width=True, type='secondary')
with col2:
    simulacion_2 = st.button("SIMULATION 2", use_container_width=True, type='secondary')
with col3:
    simulacion_3 = st.button("SIMULATION 3", use_container_width=True, type='secondary')

# Función para mostrar la barra de carga y los resultados
def mostrar_simulacion(simulacion, datos):
    if simulacion:
        # Barra de carga
        progress_bar = st.progress(0)
        for i in range(100):
            sleep(0.02)  # Simular una carga
            progress_bar.progress(i + 1)
        progress_bar.empty()  # Ocultar la barra de carga

        # Mostrar los resultados
        st.subheader("Simulation:")

        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        columns=[col1, col2, col3, col4, col5, col6]
        for j,i, valor,colunm in zip(features,metric, datos,columns):
            with colunm:
                st.write(f"{j}   : {valor if i == 'type' else f'{valor} {i}'}")

# Lógica para cada simulación
if simulacion_1:
    mostrar_simulacion(simulacion_1, sim1)
if simulacion_2:
    mostrar_simulacion(simulacion_2, sim2)
if simulacion_3:
    mostrar_simulacion(simulacion_3, sim3)


# Cargar el modelo preentrenado
@st.cache_resource  # Cachear el modelo para mejorar el rendimiento
def load_model(dir):
    try:
        with open(dir, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model('./models/model_over.pkl')
model_type = keras.models.load_model('./models/model_types.keras')
scaler = load_model('./models/scaler.pkl')

# Título de la app
st.markdown("""
    <h1 style='text-align: center; color: #bf1123; font-family: Arial, sans-serif;'>
        💡 OPERATOR PANEL 💡
    </h1>
""", unsafe_allow_html=True)

with st.sidebar:
    col1, col2= st.columns([1, 2])

    with col1:

        input_1=vertical_slider(
            label = "🌡️ Air temperature (C)",  #Optional
            key = "vert_01" ,
            height = 300, #Optional - Defaults to 300
            thumb_shape = "circle", #Optional - Defaults to "circle"
            step = 0.1, #Optional - Defaults to 1
            default_value=25 ,#Optional - Defaults to 0
            min_value= 0, # Defaults to 0
            max_value= 70, # Defaults to 10
            track_color = "white", #Optional - Defaults to Streamlit Red
            slider_color = ('black','red'), #Optional
            thumb_color= "orange", #Optional - Defaults to Streamlit Red
            value_always_visible = True ,#Optional - Defaults to False
        )
    with col2:

        input_2=vertical_slider(
            label = "🔥 Process temperature (C)",  #Optional
            key = "vert_02" ,
            height = 300, #Optional - Defaults to 300
            thumb_shape = "circle", #Optional - Defaults to "circle"
            step = 0.1, #Optional - Defaults to 1
            default_value=25 ,#Optional - Defaults to 0
            min_value= 0, # Defaults to 0
            max_value= 70, # Defaults to 10
            track_color = "white", #Optional - Defaults to Streamlit Red
            slider_color = ('red','white'), #Optional
            thumb_color= "orange", #Optional - Defaults to Streamlit Red
            value_always_visible = True ,#Optional - Defaults to False
        )
    input_3 = st.slider("⚙️ Rotational Speed (rpm)", 0.0, 5000.0, 1000.0, 1.0)
    input_4 = st.slider("🔩 Torque (Nm)", 0.0, 100.0, 40.0, 1.0)
    input_5 = st.slider("⌛ Tool Wear (min)", 0.0, 500.0, 0.0, 1.0)
    input_6 = st.radio("🛠️ Type of tool", ["L", "M", "H"], horizontal=True)

# Crear layout de columnas
col1, col2= st.columns([2, 2])


with col1:

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=input_1,
        title={"text": "🌡️ Air temperature (C)"},
        gauge={"axis": {"range": [0, 70]}, "bar": {"color": "red"}}
    ))    
    st.plotly_chart(fig2)

    # Crear el gráfico de bala (bullet chart)
    fig3 = go.Figure(go.Indicator(
        mode="number+gauge+delta",  # Modo del indicador
        gauge={
            'shape': "bullet",  # Forma de bala
            'axis': {'range': [500, 5000]},  # Rango del eje (mínimo y máximo)
            'bar': {'color': "yellow"},  # Color de la barra
        },
        delta={'reference': 1500},  # Valor de referencia para el delta
        value=input_3,  # Valor actual
        domain={'x': [0.1, 1], 'y': [0.6, 1]},  # Posición del gráfico
        title={'text': "RPM"},  # Título del gráfico
    ))

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig3)

    fig5 = go.Figure(go.Indicator(
    mode = "number",
    value = input_5,
    number = {'suffix': " Min", 'prefix': "⏱️"},
    domain = {'x': [0, 1], 'y': [0.6, 1]}))

    st.plotly_chart(fig5)


with col2:


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=input_2,
        title={"text": "🔥 Process temperature (C)"},
        gauge={"axis": {"range": [0, 70]}, "bar": {"color": "red"}}
    ))
    st.plotly_chart(fig)  
    

    # Crear el gráfico de bala (bullet chart)
    fig4 = go.Figure(go.Indicator(
        mode="number+gauge+delta",  # Modo del indicador
        gauge={
            'shape': "bullet",  # Forma de bala
            'axis': {'range': [0, 100]},  # Rango del eje (mínimo y máximo)
            'bar': {'color': "gray"},  # Color de la barra
        },
        delta={'reference': 20},  # Valor de referencia para el delta
        value=input_4,  # Valor actual
        domain={'x': [0.1, 1], 'y': [0.6, 1]},  # Posición del gráfico
        title={'text': "Nm"},  # Título del gráfico
    ))

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig4)


        # CSS personalizado para el cuadrado estrecho
    st.markdown(
        """
        <style>
        .contenedor-centrado {
            display: flex;
            justify-content: center; /* Centrar horizontalmente */
            align-items: center; /* Centrar verticalmente */
            margin-top: 0px; /* Margen superior */
        }
        .cuadrado-estrecho {
            background-color: #42423d; /* Color de fondo */
            border: 2px solid #ccc; /* Borde */
            border-radius: 10px; /* Bordes redondeados */
            padding: 10px; /* Espaciado interno */
            text-align: center; /* Centrar el texto */
            font-size: 48px; /* Tamaño de la letra */
            font-weight: bold; /* Texto en negrita */
            margin-top: 20px; /* Margen superior */
            width: 180px; /* Ancho del cuadrado */
            height: 180px; /* Alto del cuadrado */
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Mostrar el valor seleccionado en un cuadrado estrecho y centrado
    st.markdown(
        f'<div class="contenedor-centrado"><div class="cuadrado-estrecho">{input_6}</div></div>',
        unsafe_allow_html=True,
    )


col3, col4= st.columns([2, 2])
with col3:

    # CSS personalizado para el botón
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: #FF0000; /* Rojo */
            color: white; /* Texto blanco */
            font-size: 18px; /* Tamaño de la fuente */
            font-weight: bold; /* Texto en negrita */
            border-radius: 50%; /* Hace el botón redondo */
            width: 300px; /* Ancho del botón */
            height: 100px; /* Alto del botón */
            border: 4px solid #8B0000; /* Borde grueso y oscuro */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.6); /* Sombra para efecto 3D */
            transition: background-color 0.3s ease, transform 0.2s ease; /* Transiciones suaves */
        }
        .stButton button:hover {
            background-color: #CC0000; /* Rojo más oscuro al pasar el mouse */
            transform: scale(1.1); /* Efecto de agrandamiento */
        }
        .stButton button:active {
            background-color: #8B0000; /* Rojo oscuro al hacer clic */
            transform: scale(0.95); /* Efecto de hundimiento */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Crear una columna centrada
    col1, col2, col3 = st.columns([1, 1, 1])  # La columna central es más ancha

    # Mostrar el botón en la columna central
    with col2:
        submit = st.button("⚙️ SET", use_container_width=True)

# Mapeo del tipo de herramienta
tool_mapping = {"L": [1, 0, 0], "M": [0, 1, 0], "H": [0, 0, 1]}
input_6_L, input_6_M, input_6_H = tool_mapping[input_6]
with col4:

    # Ejecutar predicción cuando el usuario envíe el formulario
    if submit:
        try:
            features = np.array([[input_1+273.15, input_2+273.15, input_3, input_4, input_5, input_6_H, input_6_M, input_6_L]])
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