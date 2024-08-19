import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

# Cargar el dataset
df_filtered = pd.read_csv('dataset_proyecto.csv')
df_filtered['FECHA'] = pd.to_datetime(df_filtered['FECHA'])  # Asegúrate de que FECHA es de tipo datetime

# Cargar el modelo entrenado
model = load_model('mi_modelo_lstm.h5')

# Selección de las características
features = ['Media_Movil_5', 'RSI', 'NUMERO ACCIONES', 'VALOR EFECTIVO']
target = 'PRECIO'

# Escalar las características
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_filtered[features + [target]])

# Configuración de la interfaz
st.title('Predicción de Precios de Cierre')

# Selección de la empresa
companies = df_filtered['EMISOR'].unique()
selected_company = st.selectbox('Selecciona la empresa:', companies)

# Selección del rango de fechas
start_date = st.date_input('Fecha de inicio', value=datetime.now())
end_date = st.date_input('Fecha de fin', value=datetime.now())

# Convertir fechas seleccionadas a datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Botón para generar la predicción
if st.button('Generar Predicción'):
    # Filtrar los datos según la selección del usuario
    filtered_data = df_filtered[(df_filtered['EMISOR'] == selected_company) &
                                (pd.to_datetime(df_filtered['FECHA']) >= start_date) &
                                (pd.to_datetime(df_filtered['FECHA']) <= end_date)]
    
    if not filtered_data.empty:
        # Preprocesar datos
        filtered_data_scaled = scaler.transform(filtered_data[features + [target]])
        
        # Crear secuencias para predicción
        seq_length = 30  # Usa el mismo valor que utilizaste durante el entrenamiento
        X = []
        for i in range(seq_length, len(filtered_data_scaled)):
            X.append(filtered_data_scaled[i-seq_length:i])
        X = np.array(X)
        
        # Hacer predicción
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], len(features))), predictions)))[:, -1]
        
        # Graficar los resultados
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Precio real
        ax.plot(filtered_data['FECHA'], filtered_data['PRECIO'], label='Precio Real', color='blue')
        
        # Predicciones
        prediction_dates = filtered_data['FECHA'].iloc[seq_length:].values
        ax.plot(prediction_dates, predictions, label='Predicciones', color='orange')
        
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.set_title(f'Predicción de Precios para {selected_company}')
        ax.legend()
        
        st.pyplot(fig)
    else:
        st.write("No hay datos disponibles para el rango de fechas seleccionado.")
