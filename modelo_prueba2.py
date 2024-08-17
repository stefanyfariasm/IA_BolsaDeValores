#con los nombres de las empresas

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
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

# Reemplazar comas por puntos en las columnas de características y en la columna objetivo
data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']] = data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']].replace({',': '.'}, regex=True)

# Convertir las columnas a tipo float
data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']] = data[['VALOR NOMINAL', 'NUMERO ACCIONES', 'VALOR EFECTIVO', 'PRECIO']].astype(float)

# Limpiar los nombres de los emisores eliminando duplicados y espacios adicionales
data['EMISOR'] = data['EMISOR'].str.strip()
emisores = sorted(data['EMISOR'].unique())

# Función para predecir el precio de las acciones por empresa
def predict_stock_price_by_company(emisor, start_date, end_date):
    # Filtrar los datos por emisor
    company_data = data[data['EMISOR'] == emisor]

    # Filtrar por rango de fechas
    company_data = company_data[(company_data['FECHA'] >= start_date) & (company_data['FECHA'] <= end_date)]

    # Asegurarse de que haya suficientes datos para el entrenamiento
    if len(company_data) < 60:
        print("No hay suficientes datos para entrenar el modelo.")
        return

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

    # Invertir la normalización solo para la columna de precios predichos
    predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], predictions), axis=1))[:, -1]

    # Comparar las predicciones con los valores reales
    real_prices = company_data['PRECIO'].values[-len(y_test):]

    plt.figure(figsize=(14, 5))
    plt.plot(real_prices, color='blue', label='Precio Real')
    plt.plot(predictions, color='red', label='Precio Predicho')
    plt.title(f'Predicción de Precio de Acciones para {emisor}')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()

# Interfaz gráfica
def run_interface():
    def on_predict():
        emisor = emisor_var.get()
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        predict_stock_price_by_company(emisor, start_date, end_date)

    root = tk.Tk()
    root.title("Predicción de Precios de Acciones")

    # Emisor
    tk.Label(root, text="Selecciona el Emisor:").grid(row=0, column=0, padx=10, pady=10)
    emisor_var = tk.StringVar()
    emisor_menu = ttk.Combobox(root, textvariable=emisor_var, values=emisores)
    emisor_menu.grid(row=0, column=1, padx=10, pady=10)

    # Fecha de inicio
    tk.Label(root, text="Fecha de Inicio (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=10)
    start_date_entry = tk.Entry(root)
    start_date_entry.grid(row=1, column=1, padx=10, pady=10)

    # Fecha de fin
    tk.Label(root, text="Fecha de Fin (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=10)
    end_date_entry = tk.Entry(root)
    end_date_entry.grid(row=2, column=1, padx=10, pady=10)

    # Botón de Predicción
    predict_button = tk.Button(root, text="Predecir", command=on_predict)
    predict_button.grid(row=3, columnspan=2, padx=10, pady=20)

    root.mainloop()

# Ejecutar la interfaz gráfica
run_interface()
