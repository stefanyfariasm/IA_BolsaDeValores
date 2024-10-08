import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv', delimiter=';')

# Eliminar posibles espacios adicionales en los nombres de las columnas
data.columns = data.columns.str.strip()

# Convertir la columna 'FECHA' a formato datetime
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Reemplazar comas por puntos en las columnas de características y en la columna objetivo
data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']] = data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']].replace({',': '.'}, regex=True)

# Convertir las columnas a tipo float
data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']] = data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']].astype(float)

# Limpiar los nombres de los emisores eliminando duplicados y espacios adicionales
data['EMISOR'] = data['EMISOR'].str.strip()

# Filtrar solo las 5 empresas
empresas_interes = ['CORPORACION FAVORITA C.A.', 'BANCO GUAYAQUIL S.A.', 'BANCO DE LA PRODUCCION S.A . PRODUBANCO', 'CERVECERIA NACIONAL CN S A', 'BANCO PICHINCHA C.A.']
data = data[data['EMISOR'].isin(empresas_interes)]

# Obtener la lista de emisores filtrados
emisores = sorted(data['EMISOR'].unique())

def predict_stock_price_by_company(emisor, start_date, end_date):
    # Filtrar los datos por emisor
    company_data = data[data['EMISOR'] == emisor]

    # Ordenar por fecha
    company_data.sort_values('FECHA', inplace=True)

    # Seleccionar las columnas relevantes
    features = ['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO']
    target = 'PRECIO'

    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(company_data[features + [target]])

    # Crear un DataFrame con los datos normalizados
    scaled_df = pd.DataFrame(scaled_data, columns=features + [target])

    # Separar las características y la variable objetivo
    X = scaled_df[features].values
    y = scaled_df[target].values

    # Definir la longitud de la secuencia
    sequence_length = 30

    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Verificar que haya suficientes datos
    if len(X_seq) == 0 or len(y_seq) == 0:
        print(f"No hay suficientes datos para la empresa {emisor}.")
        return

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # Construir el modelo LSTM
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compilar el modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Entrenar el modelo
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1)

    # Filtrar los datos para predicción por rango de fechas
    prediction_data = company_data[(company_data['FECHA'] >= start_date) & (company_data['FECHA'] <= end_date)]

    # Normalizar los datos para predicción
    prediction_scaled = scaler.transform(prediction_data[features + [target]])

    # Si no hay suficientes datos para formar una secuencia en el rango seleccionado,
    # se usa la secuencia más reciente de la compañía
    if len(prediction_data) < sequence_length:
        # Usar los últimos 'sequence_length' días de datos para predicción
        prediction_seq = scaled_data[-sequence_length:, :-1]  # omitimos la última columna (target)
        prediction_dates = company_data['FECHA'][-sequence_length:]
    else:
        # Crear secuencias para la predicción desde los datos seleccionados
        prediction_seq = []
        for i in range(sequence_length, len(prediction_scaled)):
            prediction_seq.append(prediction_scaled[i-sequence_length:i, :len(features)])
        
        prediction_seq = np.array(prediction_seq)
        prediction_dates = prediction_data['FECHA'].iloc[sequence_length:]

    # Asegurarse de que tenemos la forma correcta para la predicción
    prediction_seq = np.array([prediction_seq]) if len(prediction_seq.shape) == 2 else prediction_seq

    if prediction_seq.shape[0] == 0:
        print(f"No se pueden generar secuencias para la predicción con los datos disponibles en el rango de fechas seleccionado para {emisor}.")
        return

    # Hacer predicciones sobre las fechas seleccionadas
    predictions = model.predict(prediction_seq)

    # Invertir la normalización solo para la columna de precios predichos
    predictions = scaler.inverse_transform(np.concatenate((prediction_seq[:, -1, :], predictions), axis=1))[:, -1]

    # Comparar las predicciones con los valores reales
    real_prices = prediction_data['PRECIO'].values[sequence_length:]

    # Mostrar las fechas en el gráfico
    plt.figure(figsize=(14, 5))
    plt.plot(prediction_dates, real_prices, color='blue', label='Precio Real')
    plt.plot(prediction_dates, predictions, color='red', label='Precio Predicho')
    plt.title(f'Predicción de Precio de Acciones para {emisor}')
    plt.xlabel('Fechas')
    plt.ylabel('Precio')
    plt.xticks(rotation=45)  # Rotar las fechas para mejor visualización
    plt.legend()
    plt.show()


def run_interface():
    def on_predict():
        emisor = emisor_var.get()
        start_date = pd.to_datetime(start_date_entry.get_date())
        end_date = pd.to_datetime(end_date_entry.get_date())

        predict_stock_price_by_company(emisor, start_date, end_date)

    root = tk.Tk()
    root.title("Predicción de Precios de Acciones")

    tk.Label(root, text="Selecciona el Emisor:").grid(row=0, column=0, padx=10, pady=10)
    emisor_var = tk.StringVar()
    emisor_menu = ttk.Combobox(root, textvariable=emisor_var, values=emisores)
    emisor_menu.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Fecha de Inicio:").grid(row=1, column=0, padx=10, pady=10)
    start_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
    start_date_entry.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Fecha de Fin:").grid(row=2, column=0, padx=10, pady=10)
    end_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
    end_date_entry.grid(row=2, column=1, padx=10, pady=10)

    predict_button = tk.Button(root, text="Predecir", command=on_predict)
    predict_button.grid(row=3, columnspan=2, padx=10, pady=20)

    root.mainloop()

run_interface()

