from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Загрузка данных Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Подготовка данных
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Создание модели CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучение модели
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

import matplotlib.pyplot as plt

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy on Test Data: {accuracy}")

# Визуализация результатов
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


