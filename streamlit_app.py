import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

import pathlib


# Function to load CSS from the 'assets' folder
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = pathlib.Path("assets/styles.css")
load_css(css_path)


st.title('🤖 Micro proyecto 2 Clasificacion ODS MP-68')
st.subheader( "Objetivo: Desarrollar una solución, basada en técnicas de procesamiento de lenguaje natural y machine learning, que facilite la interpretación y análisis de información textual para la identificación de relaciones semánticas con los Objetivos de Desarrollo Sostenibles.")

st.info('En esta aplicacion se puede ingresar un texto y se otorgara una clasificacion ODS.')

user_message = st.text_area("Ingresa el texto a clasificar", key="styledtextarea")
st.write("Selecciona el boton Predecir Clasificacion para visualizar el ODS")

mensaje = pd.Series(user_message, dtype="str")

#Funcion que se encarga del preprocesamiento de los textos
def convert_token(textos):
    #Crea la instancia Tokenizer que encuentra la secuencia de caracteres alfanumericos
    tokenizer = RegexpTokenizer(r'\w+')
    #Aplica el tokenizer a cada elembneto en textos, donde cada elemento es una lista de palabras sin puntuacion
    tokenized_no_punct = textos.apply(lambda x: tokenizer.tokenize(x))
    #Carga la lista predefinida de palabras vacias en espanol
    nltk_stopwords = stopwords.words("spanish")
    #Itera en la lista de palabras tokenized_no_punct y filtra las palabras presentes en nltk_stopwords
    no_stopwords = tokenized_no_punct.apply(lambda x: [token for token in x if token not in nltk_stopwords])
    #Crea instancia de stemmer, para encontrar la raiz de las palabras
    stemmer = PorterStemmer()
    #Crea una nueva lista con las palabras reducidad a su raiz
    stemmed = no_stopwords.apply(lambda x: [stemmer.stem(token) for token in x])
    #Une las palabras procesadas en cada lista de stemmed en una sola cadena de texto, separada por un espacio
    processed = stemmed.apply(lambda x: ' '.join(x))
    return processed

texto_procesado = convert_token(mensaje)

try:
    vectorizer = joblib.load('vectorizer.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    tsvd = joblib.load('tsvd.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    mejor_modelo = joblib.load('mejor_modelo.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")

def pipeline(texto_procesado):
    vector = vectorizer.transform(texto_procesado)
    svd = tsvd.transform(vector.astype('float32'))
    return svd

if st.button("Predecir Clasificacion", key="orange"):
    #if texto_procesado
    texto_svd = pipeline(texto_procesado)
    contador = np.sum(texto_svd)
    if contador == 0:
        st.write("Revisar el texto ingresado")
    else:
        y_pred = mejor_modelo.predict(texto_svd)
        st.write("### Clasificacion ODS: ", y_pred)
    

