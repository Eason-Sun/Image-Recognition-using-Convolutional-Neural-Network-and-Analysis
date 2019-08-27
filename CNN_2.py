# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt

batch_size = 32
num_classes = 10
epochs = 20
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
histories = []

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Construct CNN architecture with 3x3 filters
model_deep = Sequential()
model_deep.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model_deep.add(Activation('relu'))
model_deep.add(Conv2D(32, (3, 3), padding='same'))
model_deep.add(Activation('relu'))
model_deep.add(MaxPooling2D(pool_size=(2, 2)))
model_deep.add(Dropout(0.25))

model_deep.add(Conv2D(64, (3, 3), padding='same'))
model_deep.add(Activation('relu'))
model_deep.add(Conv2D(64, (3, 3), padding='same'))
model_deep.add(Activation('relu'))
model_deep.add(MaxPooling2D(pool_size=(2, 2)))
model_deep.add(Dropout(0.25))

model_deep.add(Flatten())
model_deep.add(Dense(512))
model_deep.add(Activation('relu'))
model_deep.add(Dropout(0.5))
model_deep.add(Dense(num_classes))
model_deep.add(Activation('softmax'))

# Construct CNN architecture with 5x5 filters
model_shallow = Sequential()
model_shallow.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model_shallow.add(Activation('relu'))
model_shallow.add(MaxPooling2D(pool_size=(2, 2)))
model_shallow.add(Dropout(0.25))

model_shallow.add(Conv2D(64, (5, 5), padding='same'))
model_shallow.add(Activation('relu'))
model_shallow.add(MaxPooling2D(pool_size=(2, 2)))
model_shallow.add(Dropout(0.25))

model_shallow.add(Flatten())
model_shallow.add(Dense(512))
model_shallow.add(Activation('relu'))
model_shallow.add(Dropout(0.5))
model_shallow.add(Dense(num_classes))
model_shallow.add(Activation('softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model_deep.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])
model_shallow.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('Not using data augmentation.')
history_deep = model_deep.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 shuffle=True)
histories.append(history_deep)
history_shallow = model_shallow.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 shuffle=True)
histories.append(history_shallow)

fig=plt.figure(figsize=(16, 14), dpi= 80, facecolor='w', edgecolor='k')
for history in histories:
    plt.plot(history.history['acc'])
plt.title('Training Accuracy', fontsize=18)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['two 3x3 filters', 'one 5x5 filter'], loc='upper left', fontsize=14)
plt.show()

fig=plt.figure(figsize=(16, 14), dpi= 80, facecolor='w', edgecolor='k')
for history in histories:
    plt.plot(history.history['val_acc'])
plt.title('Testing Accuracy', fontsize=18)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['two 3x3 filters', 'one 5x5 filter'], loc='upper left', fontsize=14)
plt.show()