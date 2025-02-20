import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from streamlit_vertical_slider import vertical_slider
from time import sleep

# Configuración de estilo
st.set_page_config(
    page_title="Milling machine performance prediction",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "https://github.com/Danielgarpra/Milling_failure_prediction_ML"
    }
)


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
model_type = load_model('./models/model_types.pkl')
scaler = load_model('./models/scaler.pkl')

# Título de la app
st.markdown("""
    <h1 style='text-align: center; color: #BEBEBE; font-family: Serif, sans-serif;font-size: 65px;'>
        💡 OPERATOR PANEL 💡
    <br>
    </h1>
""", unsafe_allow_html=True)


# Definir las métricas y simulaciones
data=[27.0,35.0,1000.0,20.0,0.0,'L']
plus_time=0
features = ["🌡️ ", "🔥 ", '⚙️ ', "🔩 ", "⌛ ", "🛠️ "]
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
        columns = [col1, col2, col3, col4, col5, col6]
        for j, i, valor, colunm in zip(features, metric, datos, columns):
            with colunm:
                st.write(f"<p style='font-size: 30px;'>{j}   : {valor if i == 'type' else f'{valor} {i}'}</p>",unsafe_allow_html=True)



# Lógica para cada simulación
if simulacion_1:
    mostrar_simulacion(simulacion_1, sim1)

if simulacion_2:
    mostrar_simulacion(simulacion_2, sim2)

if simulacion_3:
    mostrar_simulacion(simulacion_3, sim3)



with st.sidebar:
    col1, col2 = st.columns([2, 2])

    with col1:
        input_1 = vertical_slider(
            label="🌡️ Air Tª(C)",
            key="vert_01",
            height=250,
            thumb_shape="circle",
            step=0.01,
            default_value=25.0,
            min_value=20.0,
            max_value=35.0,
            track_color="white",
            slider_color=('black', 'red'),
            thumb_color="orange",
            value_always_visible=True,
        )
        input_1 = st.number_input("Air Tª(C)", min_value=20.0, max_value=35.0, value=input_1, step=0.01)
    
    with col2:
        input_2 = vertical_slider(
            label="🔥 Process Tª(C)",
            key="vert_02",
            height=250,
            thumb_shape="circle",
            step=0.01,
            default_value=35.0,
            min_value=30.0,
            max_value=45.0,
            track_color="white",
            slider_color=('black', 'red'),
            thumb_color="orange",
            value_always_visible=True,
        )
        input_2 = st.number_input("Process Tª(C)", min_value=30.0, max_value=45.0, value=input_2, step=0.01)


    input_3 = st.slider("⚙️ Rotational Speed (rpm)", 1000.0, 3000.0, 1000.0, 1.0)
    input_4 = st.slider("🔩 Torque (Nm)", 0.0, 90.0, 40.0, 0.1)
    input_5 = st.slider("⌛ Tool Wear (min)", 0.0, 400.0, 0.0, 1.0)
    input_6 = st.radio("🛠️ Type of tool", ["L", "M", "H"], horizontal=True)

# Crear layout de columnas
col1, col2, col3= st.columns([2, 2, 1])

with col1:
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=input_1,
        number={'suffix': " ºC"},
        delta={'reference': data[0]},
        title={"text": "🌡️ Air temperature"},
        gauge={"axis": {"range": [20, 35]}, "bar": {"color": "red"}}
    ))
    st.plotly_chart(fig2)

    fig3 = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        gauge={
            'shape': "bullet",
            'axis': {'range': [1000, 3000]},
            'bar': {'color': "yellow"},
        },
        delta={'reference': data[2]},
        value=input_3,
        domain={'x': [0.1, 1], 'y': [0.6, 1]},
        title={'text': "RPM"},
    ))
    st.plotly_chart(fig3)


with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=input_2,
        number={'suffix': " ºC"},
        delta={'reference': data[1]},
        title={"text": "🔥 Process temperature"},
        gauge={"axis": {"range": [30, 45]}, "bar": {"color": "red"}}
    ))
    st.plotly_chart(fig)

    fig4 = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, 90]},
            'bar': {'color': "gray"},
        },
        delta={'reference': data[3]},
        value=input_4,
        domain={'x': [0.1, 1], 'y': [0.6, 1]},
        title={'text': "Nm"},
    ))
    st.plotly_chart(fig4)

with col3:



    st.markdown(
        """
        <style>
        .contenedor-centrado {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 0px;
        }
        .cuadrado-estrecho {
            background-color: #42423d;
            border: 2px solid #ccc;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            margin-top: 240px;
            width: 100px;
            height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="contenedor-centrado"><div class="cuadrado-estrecho">{input_6}</div></div>',
        unsafe_allow_html=True,
    )

    
    fig5 = go.Figure(go.Indicator(
        mode="number",
        value=input_5,
        number={'suffix': " Min", 'prefix': "⏱️"},
        domain={'x': [0.2, 0.8], 'y': [0.2, 0.7]}
    ))
    st.plotly_chart(fig5)
    

col3, col4 = st.columns([2, 2])
with col3:
    st.markdown(
        """
        <style>
        .boton-rojo button {
            background-color: #FF0000;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 50%;
            width: 300px;
            height: 300px;
            border: 4px solid #8B0000;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.6);
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .boton-rojo button:hover {
            background-color: #CC0000;
            transform: scale(1.1);
        }
        .boton-rojo button:active {
            background-color: #8B0000;
            transform: scale(0.95);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="contenedor-centrado"><div class="boton-rojo">',
        unsafe_allow_html=True,
    )
    submit = st.button("📊 SET CONFIGURATION",type='primary' ,use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# Mapeo del tipo de herramienta
tool_mapping = {"L": [1, 0, 0], "M": [0, 1, 0], "H": [0, 0, 1]}
input_6_L, input_6_M, input_6_H = tool_mapping[input_6]

with col4:
    if submit:
        try:
            features = np.array([[input_1 + 273.15, input_2 + 273.15, input_3, input_4, input_5, input_6_H, input_6_L, input_6_M]])
            prediction = model.predict(features)[0]
            predict_proba = model.predict_proba(features)[0]

            st.markdown("### 📢 Resultado de la Predicción")
            if prediction == 0:
                st.success("✅ La fresadora **funcionará correctamente**.")
                st.success(f"Con un {predict_proba[0] * 100:.1f}% de probabilidad")
            else:
                st.error("⚠️ **¡Cuidado!** Se prevé una **falla** en la fresadora.")
                st.error(f"Con un {predict_proba[1] * 100:.1f}% de probabilidad")

                features2 = np.array([[input_1+ 273.15, input_2+ 273.15, input_3, input_4, input_5, input_6_H, input_6_L, input_6_M]])
                prediction2 = model_type.predict(scaler.transform(features2))[0]

                if prediction2[0] > 0.5:
                    st.error(f'Se va a producir fallo por desgaste ({prediction2[0]*100:.2f}%)')
                    st.success("Cambia la herramienta")
                if prediction2[1] > 0.5:
                    st.error(f'Se va a producir fallo por disipación de calor ({prediction2[1]*100:.2f}%)')
                    st.success("Controla la temperatura")
                if prediction2[2] > 0.5:
                    st.error(f'Se va a producir fallo por potencia ({prediction2[2]*100:.2f}%)')
                    st.success("Cambia la velocidad rotacional")
                if prediction2[3] > 0.5:
                    st.error(f'Se va a producir fallo por sobreesfuerzo ({prediction2[3]*100:.2f}%)')
                    st.success("Baja el par torsión o cambia la herramienta")


        except Exception as e:
            st.error(f"Error durante la predicción: {e}")