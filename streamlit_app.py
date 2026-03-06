import streamlit as st

st.title('🤖 Micro proyecto 2 Clasificacion ODS MP-68')

st.info('En esta aplicacion se puede ingresar un texto y se otorgara una clasificacion ODS.')

user_message = st.text_area("Ingresa el texto a clasificar")
st.write(f"Tu texto es: {user_message}")
