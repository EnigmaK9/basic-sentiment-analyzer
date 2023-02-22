from textblob import TextBlob

# Define un texto para analizar
texto = "Me encanta este nuevo restaurante, la comida es deliciosa"

# Crea un objeto TextBlob con el texto y analiza su sentimiento
sentimiento = TextBlob(texto).sentiment.polarity

# Imprime el resultado del análisis de sentimiento
if sentimiento > 0:
    print("El texto es positivo")
elif sentimiento < 0:
    print("El texto es negativo")
else:
    print("El texto es neutral")
