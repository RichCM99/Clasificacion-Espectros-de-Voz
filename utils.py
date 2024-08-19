import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import (Model, 
                                     load_model, 
                                     Sequential)
from tensorflow.keras import layers

from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score,
                             confusion_matrix)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def model():
    '''
    Crea y devuelve un modelo secuencial de red neuronal con varias capas densas y capas de dropout.
    
    Returns:
        model (Sequential): Modelo secuencial de Keras.
    '''

    # Definir el modelo secuencial

    model = Sequential([
        # Capa densa con 100 unidades y activación ReLU
        layers.Dense(100, activation='relu'),  
        # Capa de dropout con una tasa de 0.3
        layers.Dropout(0.3),    
        # Capa densa con 75 unidades y activación ReLU               
        layers.Dense(75, activation='relu'),   
        # Capa de dropout con una tasa de 0.3
        layers.Dropout(0.3),
        # Capa densa con 75 unidades y activación ReLU
        layers.Dense(50, activation='relu'),
        # Capa de dropout con una tasa de 0.3
        layers.Dropout(0.3),
        # Capa densa con 75 unidades y activación ReLU
        layers.Dense(35, activation='relu'),   # Capa densa con 35 unidades y activación ReLU
        # Capa de dropout con una tasa de 0.3
        layers.Dropout(0.3),
        # Capa de salida con activación lineal
        layers.Dense(1, activation='linear')   
    ])
    return model

def compile_fit_model(model, X_train, y_train, epochs=200, lr=0.0001):
    '''
    Compila y entrena el modelo secuencial con los datos de entrenamiento.
    
    Args:
        model (Sequential): El modelo secuencial de Keras a compilar y entrenar.
        X_train (numpy.ndarray): Datos de características para el entrenamiento.
        y_train (numpy.ndarray): Datos de etiquetas para el entrenamiento.
        epochs (int, optional): Número de épocas para entrenar el modelo. Por defecto es 200.
        lr (float, optional): Tasa de aprendizaje para el optimizador Adam. Por defecto es 0.0001.
    
    Returns:
        model (Sequential): El modelo entrenado.
        history (History): Objeto que contiene el historial del entrenamiento.
    '''

    # Compilar el modelo con el optimizador Adam y la función de pérdida BinaryCrossentropy

    model.compile(
        # Configurar el optimizador con la tasa de aprendizaje especificada
        optimizer=Adam(learning_rate=lr),  
        # Definir la función de pérdida
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  
        # Definir las métricas de evaluación
        metrics=['accuracy']  
    )

    # Entrenar el modelo
    history = model.fit(
        # Datos de entrada
        X_train,  
        # Etiquetas de entrada
        y_train,
        # Número de épocas
        epochs=epochs,
        # Tamaño del batch
        batch_size=3,
        # parámetro para indicar que no se imprima en consola el progreso
        verbose=0  
    )

    return model, history

def specificity_score(y_true, y_pred):
    '''
    Calcula y devuelve la especificidad (o tasa de verdaderos negativos) de las predicciones.
    
    Args:
        y_true (numpy.ndarray): Valores verdaderos de las etiquetas.
        y_pred (numpy.ndarray): Predicciones del modelo.
    
    Returns:
        float: La especificidad calculada.
    '''

    # Obtener la matriz de confusión y extraer los valores de TN (True Negatives) y FP (False Positives)
    tn, fp, _, _ = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel() 
    # Calcular la especificidad
    return tn / (tn + fp) 

def model_evaluate(model, X_test):
    '''
    Evalúa el modelo en los datos de prueba y devuelve las predicciones.
    
    Args:
        model (Sequential): El modelo secuencial de Keras a evaluar.
        X_test (numpy.ndarray): Datos de características para la evaluación.
    
    Returns:
        numpy.ndarray: Predicciones del modelo.
    '''

    # Generar predicciones del modelo

    # Obtener los logits (salidas sin activar)
    logits = model.predict(X_test, verbose=0)
    # Aplicar la función sigmoide para obtener probabilidades  
    probs = tf.squeeze(tf.nn.sigmoid(logits))
    # Convertir probabilidades en predicciones binarias (0 o 1)
    y_preds = tf.where(probs <= 0.5, 0, 1)
    # Devolver las predicciones como un array numpy

    return y_preds.numpy()  
