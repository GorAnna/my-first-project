
import tensorflow as tf
from keras.models import Sequential #для созд слоев
from keras.layers import SimpleRNN, Dense, Flatten, Input, Reshape #для типов слоев: для реализ прост RNN, плотного(полносвязн),преобр в вектор, вх дан, д изм формы дан
from keras.callbacks import ModelCheckpoint #имп колбэка кот позв сохр модель в проц обуч

import librosa #это пакет Python для анализа музыки и аудио. Он предоставляет строительные блоки для
# создания структур, которые помогают получать информацию о музыке

import numpy as np
from sklearn.model_selection import train_test_split #д разд на обуч и тест наборы


# Загрузка данных из библиотеки SpeechCommands
commands = ['go', 'left', 'right', 'stop', 'up', 'down']
num_commands = len(commands)


# Загрузка аудиоданных и их предобработка 
def load_and_preprocess_data():
    X = []
    y = []
    for command in commands:
        for i in range(1, 101):  # Загрузим по 100 примеров каждой команды
            file_path = f"path/to/SpeechCommands/{command}/{command}_{i}.wav"
            audio, _ = librosa.load(file_path, sr=None, duration=1)  # Загрузка аудиофайла
            mfccs = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13)  # Извлечение MFCC
            X.append(mfccs)
            y.append(commands.index(command))
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = load_and_preprocess_data()

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание простой RNN модели
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    SimpleRNN(units=64, activation='relu'),
    Dense(units=num_commands, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение RNN
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
