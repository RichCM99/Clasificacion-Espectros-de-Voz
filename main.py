import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import (Model, 
                                     load_model, 
                                     Sequential)
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score,
                             confusion_matrix)

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from table_data_utils import *


# auxiliar para saber en donde se está ejecutando el código (GPU o CPU)
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    print(device)

# importación de los datos
datos_pacientes = pd.read_csv("ReplicatedAcousticFeatures-ParkinsonDatabase.csv")

# se definen los subconjuntos resultantes de la selección de variables

# modelo random forest mean decrease gini subset
subset_dec_gini_vars = [
    "HNR05","HNR15","HNR25","HNR35","HNR38","MFCC0","MFCC3",
    "MFCC4","MFCC5","MFCC6","MFCC7","MFCC8","MFCC9","MFCC10",
    "MFCC11","MFCC12","Delta0","Delta1","Delta2","Delta3",
    "Delta5","Delta7","Delta9","Delta10","Delta11","Delta12"
    ]

# modelo random forest mean decrease accuracy subset
subset_dec_acc_vars = [
    "HNR05", "HNR15", "HNR25", "HNR35", "HNR38", "PPE", "MFCC3", 
    "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC8", "MFCC9", "MFCC10", 
    "MFCC11", "MFCC12", "Delta0", "Delta1", "Delta2", "Delta3", "Delta4", 
    "Delta5", "Delta9", "Delta10", "Delta11", "Delta12"
    ]
# modelo random grandient boosting importancia subset
subset_imp_xgb_vars = [
    "Delta0", "HNR38", "MFCC4", "PPE", "HNR35", "GNE", "Delta11", "MFCC5",
    "Delta5", "MFCC3", "RPDE", "MFCC10", "Shim_loc", "Shi_APQ11", "MFCC9",
    "Delta7", "MFCC2", "MFCC6", "MFCC11", "MFCC1", "DFA", "Delta12",
    "Shim_APQ5", "MFCC7", "Delta6"
    ]

# intersección de todos los subconjuntos anteriores
subset_interseccion_vars = [
    "HNR35", "HNR38", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "MFCC9", 
    "MFCC10", "MFCC11", "Delta0", "Delta5", "Delta11", "Delta12"
    ]

# variable que contiene el sexo de los pacientes
sex = np.array(datos_pacientes[['ID', 'Gender']].drop_duplicates()['Gender'])

# se almacenan los id's de cada paciente
ids = datos_pacientes['ID'].to_numpy()

# se quitan las variables ID, Recording, Status y Gender del conjunto de var independientes
X = datos_pacientes.drop(columns=['ID', 'Recording', 'Status', 'Gender'])

# se selecciona el subconjunto con el que vamos a entrenar el modelo
X = X[subset_dec_gini_vars]

# se guardan las etiquetas reales
y = np.array(datos_pacientes['Status'])

# se define el método de validación (leave one group out)
logo = LeaveOneGroupOut()

# se indica que dejaremos las observaciones de un paciente en el conjunto de prueba
logo.get_n_splits(X, y, groups=ids)

# para almacenar las etiquetas reales de los pacientes
y_group_true = np.zeros(80)

# para almacenar las etiquetas predichas de los pacientes
y_group_pred = np.zeros(80)

# proceso de validación
for i, (train_index, test_index) in enumerate(logo.split(X, y, groups=ids)):

    # por cada iteración se genera una semilla
    tf.random.set_seed(i)

    # se definen conjunto de entrenamiento y de prueba
    X_train = X.iloc[train_index, :]; X_test = X.iloc[test_index, :]

    # se definen etiquetas de entrenamiento y de prueba
    y_train = y[train_index]; y_test = y[test_index]

    # se define el escalador que utilizaremos sobre los datos (MinMaxScaler)
    scaler = MinMaxScaler()

    # se ajustan y almacenan los parámetros de escalamiento sobre el conjunto de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)

    # se escalan los datos del conjunto de prueba con los parámetros encontrados del 
    # conjunto de entrenamiento
    X_test_scaled = scaler.transform(X_test)

    # se define la arquitectura del modelo a utilizar
    modelo = model()

    # compilación y entrenamiento del modelo definido anteriormente
    trained_model, _ = compile_fit_model(modelo, X_train=X_train_scaled, y_train=y_train)

    # evaluación del modelo utilizando la función sigmoide y un umbral de 0.5
    y_test_preds =  model_evaluate(model=trained_model, X_test=X_test_scaled)

    # voto mayoritario para clasificación de los pacientes - predicciones
    y_group_pred[i] = np.argmax(np.bincount(y_test_preds, minlength = 2))

    # extracción de etiqueta real de cada paciente
    y_group_true[i] = np.argmax(np.bincount(y_test, minlength = 2))


    print(f"********************** iteracion {i + 1} **********************")
    
    # limpiamos la sesion
    del(modelo, trained_model, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)
    
    # se interrumpen los procesos internos que usa tensorflow
    tf.keras.backend.clear_session()


# se imprimen en consola las métricas de evaluación

# accuracy
print(accuracy_score(y_pred=y_group_pred, y_true=y_group_true))

# precision
print(precision_score(y_pred=y_group_pred, y_true=y_group_true))

# recall
print(recall_score(y_pred=y_group_pred, y_true=y_group_true))

# specificity
print(specificity_score(y_pred=y_group_pred, y_true=y_group_true))


# se imprimen las mismas métricas de evaluación por sexo
hombres = y_group_true[np.where(sex == 0)]
mujeres = y_group_true[np.where(sex == 1)]

hombres_preds = y_group_pred[np.where(sex == 0)]
mujeres_preds = y_group_pred[np.where(sex == 1)]

# accuracy
print(accuracy_score(y_pred=hombres_preds, y_true=hombres))
print(accuracy_score(y_pred=mujeres_preds,  y_true=mujeres))

# precision
print(precision_score(y_pred=hombres_preds, y_true=hombres))
print(precision_score(y_pred=mujeres_preds, y_true=mujeres))

# recall
print(recall_score(y_pred=hombres_preds, y_true=hombres))
print(recall_score(y_pred=mujeres_preds, y_true=mujeres))

# specificity
print(specificity_score(y_pred=hombres_preds, y_true=hombres))
print(specificity_score(y_pred=mujeres_preds, y_true=mujeres))
