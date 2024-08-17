import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('dataset.csv', delimiter=';')

# Eliminar posibles espacios adicionales en los nombres de las columnas
data.columns = data.columns.str.strip()

# Convertir la columna 'FECHA' a formato datetime
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Ordenar por fecha
data.sort_values('FECHA', inplace=True)

# Seleccionar las columnas relevantes
features = ['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO']
target = 'PRECIO'

# Reemplazar comas por puntos en las columnas de características y en la columna objetivo
data[features] = data[features].replace({',': '.'}, regex=True)
data[target] = data[target].replace({',': '.'}, regex=True)

# Convertir las columnas a tipo float
data[features] = data[features].astype(float)
data[target] = data[target].astype(float)  # Asegurar que la columna objetivo también sea float

# Verificar si hay valores no numéricos o faltantes
for col in features + [target]:
    invalid_rows = data[pd.to_numeric(data[col], errors='coerce').isna()]
    if not invalid_rows.empty:
        print(f"Valores no numéricos o faltantes encontrados en la columna '{col}':")
        print(invalid_rows)

# Eliminar filas con valores faltantes en las características o en la columna objetivo
data.dropna(subset=features + [target], inplace=True)

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Crear un DataFrame con los datos normalizados
scaled_df = pd.DataFrame(scaled_data, columns=features)

# Añadir la columna objetivo (PRECIO)
scaled_df[target] = data[target].values

# Separar las características y la variable objetivo
X = scaled_df[features].values
y = scaled_df[target].values

# Definir la longitud de la secuencia
sequence_length = 60

X_seq = []
y_seq = []

for i in range(sequence_length, len(X)):
    X_seq.append(X[i-sequence_length:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Verificar las formas y tipos de datos
print(X_train.shape, X_train.dtype)
print(X_test.shape, X_test.dtype)
print(y_train.shape, y_train.dtype)
print(y_test.shape, y_test.dtype)

# Construir el modelo LSTM
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=32, epochs=50)

# Hacer predicciones sobre los datos de prueba
predictions = model.predict(X_test)

# Invertir la normalización para obtener los valores originales
# Nota: Para invertir la normalización correctamente, necesitas aplicar la inversa de la escala solo a las predicciones
# y no a los valores reales, ya que estos no están escalados en el mismo DataFrame.
# El código para revertir la escala de las predicciones es incorrecto si sólo se escala parte de los datos.
# Es más adecuado simplemente visualizar la comparación sin invertir la escala si los datos están normalizados.

# Comparar las predicciones con los valores reales
real_prices = data['PRECIO'][len(data) - len(y_test):].values

plt.figure(figsize=(14, 5))
plt.plot(real_prices, color='blue', label='Precio Real')
plt.plot(predictions, color='red', label='Precio Predicho')
plt.title('Predicción de Precio de Acciones')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()
plt.show()
