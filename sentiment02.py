import random
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Descarga los datos de ejemplo de críticas de películas
nltk.download('movie_reviews')

# Carga los datos de entrenamiento y prueba
documentos = [(list(movie_reviews.words(fileid)), categoria)
             for categoria in movie_reviews.categories()
             for fileid in movie_reviews.fileids(categoria)]
random.shuffle(documentos)

conjunto_de_entrenamiento = documentos[:1600]
conjunto_de_prueba = documentos[1600:]

# Define una función para extraer características de los datos de entrada
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Extrae las características más relevantes de los datos de entrenamiento
todos_los_palabras = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(todos_los_palabras)[:2000]
training_features = nltk.classify.apply_features(extract_features, conjunto_de_entrenamiento)

# Entrena un modelo de clasificación de sentimientos utilizando Naive Bayes
sentiment_classifier = NaiveBayesClassifier.train(training_features)

# Evalúa la precisión del modelo con los datos de prueba
test_features = nltk.classify.apply_features(extract_features, conjunto_de_prueba)
print('Precisión: ', accuracy(sentiment_classifier, test_features))

# Utiliza el modelo para analizar el sentimiento de un texto dado
texto = "Esta película es realmente aburrida y predecible"
sentimiento = sentiment_classifier.classify(extract_features(texto.split()))

# Imprime el resultado del análisis de sentimiento
if sentimiento == 'pos':
    print("El texto es positivo")
else:
    print("El texto es negativo")
