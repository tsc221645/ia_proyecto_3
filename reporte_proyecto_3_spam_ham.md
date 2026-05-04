# Proyecto 3 - Clasificador SPAM/HAM con Bayes

## 1. Objetivo

El objetivo es construir un filtro SPAM/HAM usando probabilidad y Bayes. El proyecto toma como base el laboratorio 6: carga el dataset `spam_ham.csv`, realiza EDA, limpia texto, entrena un clasificador bayesiano y mide su rendimiento en una division 80% entrenamiento y 20% prueba.

## 2. EDA

El dataset tiene 5565 mensajes SMS: 4819 ham y 746 spam. La clase ham representa aproximadamente 86.6% de los datos y spam 13.4%, por lo que el conjunto esta desbalanceado. Esto afecta las metricas: accuracy puede verse alta aunque el modelo pierda mensajes spam, por eso se reportan precision, recall y F1.

Los mensajes spam suelen contener terminos promocionales o de accion como `claim`, `www`, `prize`, `tone`, `guarante`, `uk`, `rington`, `mob`, `award` y `nokia`. Los mensajes ham tienen lenguaje mas conversacional y cotidiano. La longitud promedio tambien ayuda: spam tiende a ser mas largo por incluir instrucciones, premios, enlaces o codigos.

## 3. Limpieza de datos

La limpieza aplicada fue:

- Normalizacion de etiquetas (`spam` y `ham`) y eliminacion de comillas extra.
- Relleno de mensajes vacios con cadena vacia.
- Conversion del texto a minusculas.
- Tokenizacion con `wordpunct_tokenize` de NLTK.
- Eliminacion de signos, numeros y tokens no alfabeticos mediante expresion regular `[a-z]+`.
- Eliminacion de stopwords en ingles.
- Stemming con `PorterStemmer`.

Esto reduce ruido y agrupa variantes como `claim`, `claimed` o `claiming` en una raiz comun. La desventaja es que se pierden algunos numeros telefonicos o codigos que podrian ser predictivos en spam; sin embargo, para el alcance del proyecto, el vocabulario resultante queda mas estable y explicable.

## 4. Modelo

Se usa un modelo bayesiano por presencia de palabras. Para cada palabra W se calcula:

`P(S|W) = P(W|S)P(S) / (P(W|S)P(S) + P(W|H)P(H))`

Donde:

- `P(S)` es la proporcion de mensajes spam en entrenamiento.
- `P(H)` es la proporcion de mensajes ham en entrenamiento.
- `P(W|S)` es la proporcion de mensajes spam que contienen W.
- `P(W|H)` es la proporcion de mensajes ham que contienen W.

Para evitar probabilidades cero se usa suavizado de Laplace:

`P(W|S) = (docs_spam_con_W + 1) / (total_docs_spam + 2)`

La probabilidad final de spam para un texto con palabras `W1...Wn` se calcula con la formula del anexo:

`P(S|W1...Wn) = (P1 P2 ... Pn) / ((P1 P2 ... Pn) + (1-P1)(1-P2)...(1-Pn))`

En la implementacion se usan logaritmos para evitar underflow numerico cuando el texto tiene muchas palabras.

## 5. Pruebas de rendimiento

Se uso `train_test_split` con 80% entrenamiento y 20% prueba, estratificando por clase. El conjunto final fue de 4452 mensajes para entrenamiento y 1113 para prueba. Los priors aprendidos fueron `P(spam)=0.1341` y `P(ham)=0.8659`.

| Threshold | Accuracy | Precision | Recall | F1 | TP | FN | FP | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9425 | 0.7673 | 0.8188 | 0.7922 | 122 | 27 | 37 | 927 |
| 0.2 | 0.9560 | 0.8731 | 0.7852 | 0.8269 | 117 | 32 | 17 | 947 |
| 0.3 | 0.9587 | 0.9187 | 0.7584 | 0.8309 | 113 | 36 | 10 | 954 |
| 0.4 | 0.9623 | 0.9496 | 0.7584 | 0.8433 | 113 | 36 | 6 | 958 |
| 0.5 | 0.9614 | 0.9649 | 0.7383 | 0.8365 | 110 | 39 | 4 | 960 |
| 0.6 | 0.9623 | 0.9820 | 0.7315 | 0.8385 | 109 | 40 | 2 | 962 |
| 0.7 | 0.9614 | 0.9818 | 0.7248 | 0.8340 | 108 | 41 | 2 | 962 |
| 0.8 | 0.9614 | 0.9907 | 0.7181 | 0.8327 | 107 | 42 | 1 | 963 |
| 0.9 | 0.9596 | 0.9906 | 0.7047 | 0.8235 | 105 | 44 | 1 | 963 |

El mejor threshold por F1 fue 0.4, con F1 de 0.8433, precision de 0.9496 y recall de 0.7584.

## 6. Discusion

El modelo es conservador cuando el threshold sube: aumenta precision porque casi todo lo que marca como spam realmente lo es, pero baja recall porque deja pasar mas spam como ham. Para un filtro de mensajes, esta decision puede ser deseable si se quiere evitar bloquear mensajes legitimos. Si el objetivo fuera capturar mas spam aunque haya mas falsos positivos, convendria usar un threshold menor, por ejemplo 0.2.

La limpieza mejora la interpretabilidad del vocabulario y reduce palabras poco utiles. Sin embargo, eliminar numeros puede quitar senales importantes como telefonos cortos, codigos de premio o URLs con numeros. Una mejora futura seria crear variables adicionales para detectar enlaces, numeros telefonicos, mayusculas, signos de exclamacion y longitud del mensaje.

## 7. Modulo para presentacion

El archivo `spam_ham_bayes_project.py` incluye `classify_prompt(text, threshold)`. Esta funcion devuelve:

- probabilidad de que el texto sea spam;
- prediccion `spam` o `ham`;
- threshold usado;
- las 3 palabras del prompt con mayor poder predictivo hacia spam.
