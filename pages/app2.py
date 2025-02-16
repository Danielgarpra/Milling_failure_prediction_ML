import streamlit as st
import plotly.graph_objects as go

# Título de la aplicación
st.title("Indicador de Temperatura")

# CSS personalizado para hacer el slider vertical
st.markdown(
    """
    <style>
    /* Rotar el slider verticalmente */
    div[data-testid="stSlider"] > div {
        flex-direction: column;
        align-items: center;
    }
    div[data-testid="stSlider"] > div > div {
        writing-mode: bt-lr; /* Rotar el slider */
        transform: rotate(270deg); /* Rotar 270 grados */
        padding: 50px 0; /* Ajustar el padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Slider para la temperatura
temperatura = st.slider(
    "Selecciona la temperatura", 
    min_value=-20, 
    max_value=50, 
    value=25
)

# Crear un gráfico de barra vertical
fig = go.Figure(go.Bar(
    x=[""],  # Etiqueta vacía para centrar
    y=[temperatura],
    text=[f"{temperatura}°C"],  # Mostrar el valor de la temperatura
    textposition='auto',  # Posición automática del texto
    marker_color='red',   # Color de la barra
    width=[0.5]           # Ancho de la barra
))

# Configurar el diseño del gráfico
fig.update_layout(
    yaxis=dict(range=[-20, 50], title="Temperatura (°C)"),  # Rango del eje Y
    xaxis=dict(showticklabels=False),  # Ocultar etiquetas del eje X
    height=500  # Altura del gráfico
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)