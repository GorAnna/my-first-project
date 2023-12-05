import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#библиотека бфзы выборок MNIST
from keras.datasets import mnist 


#Слой Dense состоит из units нейронов, 
#соединённых синапсами с элементами входного тензора по его последнему индексу
#Слой Flatten также не имеет параметров для обучения, но меняет форму тензора. 
# Задача этого слоя состоит в преобразовании многомерного входного тензора в одномерный тензор

from keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data() #загр. обуч. и тест. выборок

#нормализац входн. данных (для диапазона от 0 до 1)
x_train = x_train/255
x_test = x_test/255

#преобраз. выходных значений в векторы по категориям (если 5 то векторн [0,0,0,0,0,1,0,0,0,0])
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

#отображение первых 25 изображений обучающ выборки

plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()


#28x28=784 ыходами (пикс)
#128 нейронов
#10 нейронов вых.слоя
model = keras.Sequential([
    Flatten (input_shape=(28, 28, 1)),
    Dense(129, activation = 'relu'),
    Dense(10, activation = 'softmax')
    ])

print(model.summary()) #вывод структыры НС в консоль

#компиляция НС с оптимизац по Adam и критерием - категориальная кросс-энтропия
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  #ф-я потерь, критерий кач-ва при задачах классификации
              metrics=['accuracy']) #список метрик д.мониторинга

#запуск процесса обучения 80% - обуч выборка, 20% - выборка валидации 
model.fit(x_train, y_train_cat, batch_size=32, epochs=1, validation_split=0.2)
        #вх.мн-во, выход зн-я, посл кажд 32изобр ,кол обучений, обуч выборка-20%
#                              будут ред весов коэф             из всех

#проверка распознования цифр
n=150
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )
print(f"распознанная цифра: {np.argmax(res)}")

#тестовая выборка
model.evaluate(x_test, y_test_cat)

n=0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"распознанная цифра: {np.argmax(res)}") #возвращ индекс макс значения

