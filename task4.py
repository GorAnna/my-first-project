from keras.datasets import cifar10
import sklearn.model_selection
from sklearn.model_selection import train_test_split


# Загрузка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Разделение данных на тренировочный и тестовый наборы
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("Размер тренировочного набора:", x_train.shape)
print("Размер валидационного набора:", x_val.shape)
print("Размер тестового набора:", x_test.shape)

#нормализац входн. данных
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train, num_classes=10)
y_val_one_hot = to_categorical(y_val, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

import tensorflow as tf
from tensorflow import keras

#сверточн НС с неск светочн слоями, слоями подвыборки, плотными сл и выходн слоем
model = keras.Sequential()

# Свёрточный слой
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

# Слой подвыборки (пулинг) - разбиение карты признаков на более мелкие матрицы, нахождение их максимальных элементов, 
# т. е. происходит увеличение «глубины» значений
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Свёрточный слой
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Слой подвыборки (пулинг)
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Преобразование двумерного массива пикселей в одномерный
model.add(keras.layers.Flatten())

# Плотный слой
model.add(keras.layers.Dense(units=128, activation='relu'))

# Выходной слой
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

 

#обучен сверточн НС на тренировочных данных с использованием метода fit()
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

# Оценка производительности модели
score = model.evaluate(x_test, y_test_one_hot)
print("Loss:", score[0])
print("Accuracy:", score[1])
