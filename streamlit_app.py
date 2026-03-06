import streamlit as st
import pandas as pd
import nummpy as np
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

st.title('🤖 Micro proyecto 2 Clasificacion ODS MP-68')

st.info('En esta aplicacion se puede ingresar un texto y se otorgara una clasificacion ODS.')

user_message = st.text_area("Ingresa el texto a clasificar")
st.write(f"Tu texto es: {user_message}")

st.write("El tipo de mensaje es ", type(user_message))
mensaje = pd.Series(user_message, dtype="str")
st.write("El tipo de mensaje convertido es ", type(mensaje))
