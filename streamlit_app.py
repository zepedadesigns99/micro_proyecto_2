import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

st.title('🤖 Micro proyecto 2 Clasificacion ODS MP-68')

st.info('En esta aplicacion se puede ingresar un texto y se otorgara una clasificacion ODS.')

user_message = st.text_area("Ingresa el texto a clasificar")
st.write(f"Tu texto es: {user_message}")

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

st.write("Texto procesado ", texto_procesado)

"""
try:
    vectorizer = joblib.load('E:/Imagenes/vectorizer.pkl') 
    st.write("Vectorizer", vectorizer)
except FileNotFoundError:
    st.error("Model file 'vectorizer.pkl' not found. Please ensure it is in the correct directory.")

try:
    tsvd = joblib.load('E:/Imagenes/tsvd.pkl') #
except FileNotFoundError:
    st.error("Model file 'tsvd.pkl' not found. Please ensure it is in the correct directory.")

try:
    best_model = joblib.load('E:/Imagenes/best_model.pkl') #
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the correct directory.")
"""
vectorizer = pickle.load(open("E:/Imagenes/vectorizer.pkl", "rb"))
tsvd = pickle.load(open("E:/Imagenes/tsvd.pkl", "rb"))
best_model = pickle.load(open("E:/Imagenes/best_model.pkl", "rb"))

#st.write("Vectorizer", vectorizer)
