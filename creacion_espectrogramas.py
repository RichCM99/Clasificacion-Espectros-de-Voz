import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# creación de espectrogramas mfcc

# datos pacientes
pacientes_data = pd.read_csv("datos de pacientes.csv", dtype=object, index_col=None)
# pacientes_data['ubicacion'][0].split("\\")[-2]
pacientes_data['names'] = list(map(lambda x:"MFCC\\" + x.split("\\")[-2] + "\\" + x.split("\\")[-1][:-4] + ".png", pacientes_data['ubicacion']))
pacientes_data.tail()

# creacion de espectros mfcc
for i, row in pacientes_data.iterrows():
    y, sr = librosa.load(row['ubicacion'], sr=None)
    # D = librosa.stft(y, n_fft=4096, hop_length=512, win_length=4096, window='hann')
    D = librosa.feature.mfcc(y=y, sr=sr)
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.savefig(row['names'], bbox_inches='tight', pad_inches = 0)
    plt.close()


# creacion de archivo .csv para la facil lectura de los datos
# guardamos la ubicación completa de cada imágen
samples = []

main_path = os.getcwd() + '\\MFCC'

labels_dict = {'Control' : 0, 'Parkinson': 1}

for label in labels_dict.keys():
    for data in os.listdir(f"{main_path}\\{label}"):
        samples.append(f"{main_path}\\{label}\\{data}")
    for data in os.listdir(f"{main_path}\\{label}"):
        samples.append(f"{main_path}\\{label}\\{data}")    

labels = list(map(lambda x: x.split("\\")[-2], samples))

# creamos el dataframe correspondiente
dataset = pd.DataFrame(columns=['ubicacion', 'label'])
dataset['ubicacion'] = samples
dataset['label'] = [labels_dict[lab] for lab in labels]
dataset['repeticion'] = list(map(lambda x: x.split("-")[-1][6:8], samples))
dataset['paciente'] = list(map(lambda x: x.split("-")[-1][2:4], samples))
dataset['id'] = dataset['label'].astype(str) + dataset['paciente'].astype(str)

dataset.to_csv('mfcc pacientes.csv')

### creación de espectrogramas de Mel

# datos pacientes
pacientes_data = pd.read_csv("datos de pacientes.csv", dtype=object, index_col=None)
# pacientes_data['ubicacion'][0].split("\\")[-2]
pacientes_data['names'] = list(map(lambda x:"MFCC\\" + x.split("\\")[-2] + "\\" + x.split("\\")[-1][:-4] + ".png", pacientes_data['ubicacion']))
pacientes_data.tail()

# creacion de espectrogramas de mel
for i, row in pacientes_data.iterrows():
    y, sr = librosa.load(row['ubicacion'], sr=None)
    # D = librosa.stft(y, n_fft=4096, hop_length=512, win_length=4096, window='hann')
    D = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.savefig(row['names'], bbox_inches='tight', pad_inches = 0)
    plt.close()


# creacion de archivo .csv para la facil lectura de los datos
# guardamos la ubicación completa de cada imágen
samples = []

main_path = os.getcwd() + '\\MelSpec'

labels_dict = {'Control' : 0, 'Parkinson': 1}

for label in labels_dict.keys():
    for data in os.listdir(f"{main_path}\\{label}"):
        samples.append(f"{main_path}\\{label}\\{data}")
    for data in os.listdir(f"{main_path}\\{label}"):
        samples.append(f"{main_path}\\{label}\\{data}")    

labels = list(map(lambda x: x.split("\\")[-2], samples))

# creamos el dataframe correspondiente
dataset = pd.DataFrame(columns=['ubicacion', 'label'])
dataset['ubicacion'] = samples
dataset['label'] = [labels_dict[lab] for lab in labels]
dataset['repeticion'] = list(map(lambda x: x.split("-")[-1][6:8], samples))
dataset['paciente'] = list(map(lambda x: x.split("-")[-1][2:4], samples))
dataset['id'] = dataset['label'].astype(str) + dataset['paciente'].astype(str)

dataset.to_csv('mfcc pacientes.csv')

