import streamlit as st

# CSS para establecer la imagen de fondo
def set_background_image(image_url):
    css = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# URL de la imagen de fondo (puede ser una URL en línea o una ruta local)
image_url = "https://d100mj7v0l85u5.cloudfront.net/s3fs-public/IM_PenIron_05.gif"  # Cambia esto por la URL de tu imagen

# Llama a la función para establecer la imagen de fondo
set_background_image(image_url)

# Contenido de tu aplicación
st.title("PREDICTIVE CONTROL / MILLING MACHINE")
st.markdown(
    """
    <hr style="height: 10px; background-color: #fcfcfc; border: none; margin: 20px 0;">
    """,
    unsafe_allow_html=True
)