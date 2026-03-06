import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
#import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
#import seaborn as sns

import warnings # Ignorar las warnings
warnings.filterwarnings("ignore")
import joblib

#Carga de datos del archivo
data_raw = pd.read_excel('E:/Train_textos.xlsx')
#Se crea una duplicado del dataframe original
data = data_raw.copy()

x = data.drop('ODS', axis=1)
y = data['ODS']

#Separacion en set de entrenamiento y prueba, se usa stratify=variable objetivo para que se separen los datos en proporciones iguales en train y test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Convierte el DataFrame (n, 1) a Series para que funcione la funcion de reducion de la dimensionalidad TruncatedSVD
x_train = x_train.stack()
x_test = x_test.stack()

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

vectorizer = TfidfVectorizer(
    max_features=15000,  # Limita el tamaño del vocabulario
    min_df=5,            # Remueve palabras raras, terminos con menos de 5 apariciones en el corpus
    max_df=0.8,          # Remueve palabras extremadamente comunes, ignora terminos que aparezcan en mas del 80% de los textos
    ngram_range=(1, 1),  # Solo palabras individuales (unigrams)
    sublinear_tf=True,   # Reduce el impacto de palabras que aparecen muchas veces en el corpus
    use_idf=True         # Activa la frecuencia inversa (IDF), si una palabra aparece muchas veces su peso disminuye
)

#Se selecciona la reducion de dimensionalidad a 100 caracteristicas, se probo con mas caracteristicas pero el incremento en las metricas era minimo 
#1-2% de aumento con n_components = 200 y 300
tsvd = TruncatedSVD(n_components=100, n_iter=5, algorithm='randomized', random_state=42)

def pipeline(X, counter, vectorizer, tsvd):
    if counter == 0:
        processed_data = convert_token(X)
        print("\n\nTextos procesados\n", processed_data)
        vector = vectorizer.fit_transform(processed_data)
        print("\n\nTransformacion numerica de textos\n", vector)
        svd = tsvd.fit_transform(vector.astype('float32'))
        counter += 1
    else:
        processed_data = convert_token(X)
        vector = vectorizer.transform(processed_data)
        svd = tsvd.transform(vector.astype('float32'))
        
    return svd, counter, vectorizer, tsvd

counter = 0 #Para contar la primera instancia y hacer fit_transform al set de entrenamiento
#Se pasa tanto x_train como x_test por la funcion pipeline
x_train_pipe, counter, vectorizer, tsvd = pipeline(x_train, counter, vectorizer, tsvd)
x_test_pipe, counter, vectorizer, tsvd = pipeline(x_test, counter, vectorizer, tsvd)

#Se crea la instancia kfold para validacion cruzada, con 5 splits 
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

"""
    Taller Semana 6: XGBoost
    Machine Learning Supervisado
    Codigo creado por: Daniel Felipe Lopez

    Codigo modificado por: Israel Francisco Sanchez Zepeda
"""
def report_best_scores(results, n_top):
    # Esta función espera una instancia de resultados de búsqueda de cross validation, por ejemplo: search.cv_results_
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_accuracy'] == i)
        for candidate in candidates:
            print(candidate)
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_accuracy'][candidate],
                  results['std_test_accuracy'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
    results = pd.DataFrame(results)
    # Retorna los parámetros del mejor modelo basado en accuracy.
    return list(results.sort_values("rank_test_accuracy")['params'])[0]
""""
def see_results(results, n_top):
    # Esta función espera una instancia de pandas dataframe de los resultados de búsqueda de cross validation, por ejemplo: pd.DataFrame(search.cv_results_)
    #Muestra el n_top de resultados de acuerdo a accuracy
    display(results[results.columns.drop(list(results.filter(regex='split')))].sort_values("rank_test_f1")[:n_top]) 
"""
"""
    Cgnorthcutt (2018, 25nd December). Compare multiple algorithms with sklearn pipeline. Stack Overflow. 
    Obtenido el 3 de marzo del 2026 de:
    https://stackoverflow.com/questions/51695322/compare-multiple-algorithms-with-sklearn-pipeline
"""
class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=SGDClassifier()):
        """
        El estimador puede cambiar de acuerdo a diferentes clasificadores.
        En este proyecto se usara SGDClassifier, LogisticRegression y SVC
        """
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

pipe_clf = Pipeline([
    ('clf', ClfSwitcher()), #Se pasa en el pipeline la clase que permitira cambiar de estimador
])


#Parametros a buscar por RandomizedSearchCV, con 3 modelos diferentes
parameters = [
    {
        'clf__estimator': [SGDClassifier(random_state=42)], 
        'clf__estimator__penalty': ('l2', 'l1'),
        'clf__estimator__max_iter': [50, 80],
        'clf__estimator__loss': ['hinge', 'log_loss', 'modified_huber'],
    },
    {
        'clf__estimator': [SVC(random_state=42, probability=True)],
        'clf__estimator__C': [0.01, 0.1, 1, 10, 100],
        'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'clf__estimator__class_weight': ['balanced', None],
        
    },
    {
        'clf__estimator': [LogisticRegression(random_state=42)],
        'clf__estimator__l1_ratio': [1], #penalty ='l1'
        'clf__estimator__solver': ['saga'],
        'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__estimator__class_weight': ['balanced', None],
    },
    {
        'clf__estimator': [LogisticRegression(random_state=42)],
        'clf__estimator__l1_ratio': [0], #penalty ='l2'
        'clf__estimator__solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__estimator__class_weight': ['balanced', None],
    },
    {
        'clf__estimator': [LogisticRegression(random_state=42)],
        'clf__estimator__solver': ['saga'],
        'clf__estimator__l1_ratio': [0.1, 0.5, 0.9], #penalty='elasticnet'
        'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__estimator__class_weight': ['balanced', None],
    },
]

random_search = RandomizedSearchCV(
    estimator=pipe_clf, #clase que cambia de modelos de forma dinamica
    param_distributions=parameters,  #Rango de hiperparametros a explorar y modelos
    scoring=scoring, # accuracy, precision, recall, f1
    cv=kfold, # 5-fold cross-validation
    verbose=1,
    n_iter = 50, #50 combinaciones aleatorias de todos los modelos
    refit='f1', #cuando se terminen las n_iter se queda con el mejor estimador en la metrica f1
    n_jobs=-1
)

#Se realiza el fit con random_search
random_search.fit(x_train_pipe, y_train)

print("Mejores parametros de random search ", random_search.best_params_)
results = random_search.cv_results_

n_top = 10 #Selecciona en n_top de mejores modelos encontrados por random_search
best_result = report_best_scores(results, n_top)

results_df = pd.DataFrame(results)
#see_results(results_df, n_top)


best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test_pipe)
print(classification_report(y_test, y_pred))


joblib.dump(vectorizer, "vectorizer.joblib")

joblib.dump(tsvd, "tsvd.joblib")

joblib.dump(best_model, "best_model.joblib")




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

with st.form('user_input'):
    user_message = st.text_area("Ingresa el texto a clasificar")
    st.write(f"Tu texto es: {user_message}")
    st.write("Seleccionar el button Submit para empezar la clasificacion del texto ingresado")
    button = st.form_submit_button()

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

try:
    vectorizer = joblib.load('E:/Imagenes/vectorizer.joblib')
except Exception as e:
    st.error(f"Error loading model: {e}")
    
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
#vectorizer = pickle.load(open("E:/Imagenes/vectorizer.pkl", "rb"))
#tsvd = pickle.load(open("E:/Imagenes/tsvd.pkl", "rb"))
#best_model = pickle.load(open("E:/Imagenes/best_model.pkl", "rb"))

#st.write("Vectorizer", vectorizer)
