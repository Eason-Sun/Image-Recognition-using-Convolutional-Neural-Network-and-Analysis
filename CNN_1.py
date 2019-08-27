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

# Construct CNN architecture
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model_cnn.add(Activation('relu'))
model_cnn.add(Conv2D(32, (3, 3), padding='same'))
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(64, (3, 3), padding='same'))
model_cnn.add(Activation('relu'))
model_cnn.add(Conv2D(64, (3, 3), padding='same'))
model_cnn.add(Activation('relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())
model_cnn.add(Dense(512))
model_cnn.add(Activation('relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(num_classes))
model_cnn.add(Activation('softmax'))

# Construct fully connected NN architecture with 0 hidden layer
model_0_hl = Sequential()
model_0_hl.add(Flatten())
model_0_hl.add(Dense(num_classes))
model_0_hl.add(Activation('softmax'))

# Construct fully connected NN architecture with 1 hidden layer
model_1_hl = Sequential()
model_1_hl.add(Flatten())
model_1_hl.add(Dense(512))
model_1_hl.add(Activation('relu'))
model_1_hl.add(Dropout(0.5))
model_1_hl.add(Dense(num_classes))
model_1_hl.add(Activation('softmax'))

# Construct fully connected NN architecture with 2 hidden layer
model_2_hl = Sequential()
model_2_hl.add(Flatten())
model_2_hl.add(Dense(512))
model_2_hl.add(Activation('relu'))
model_2_hl.add(Dropout(0.5))
model_2_hl.add(Dense(512))
model_2_hl.add(Activation('relu'))
model_2_hl.add(Dropout(0.5))
model_2_hl.add(Dense(num_classes))
model_2_hl.add(Activation('softmax'))

# Construct fully connected NN architecture with 3 hidden layer
model_3_hl = Sequential()
model_3_hl.add(Flatten())
model_3_hl.add(Dense(512))
model_3_hl.add(Activation('relu'))
model_3_hl.add(Dropout(0.5))
model_3_hl.add(Dense(512))
model_3_hl.add(Activation('relu'))
model_3_hl.add(Dropout(0.5))
model_3_hl.add(Dense(512))
model_3_hl.add(Activation('relu'))
model_3_hl.add(Dropout(0.5))
model_3_hl.add(Dense(num_classes))
model_3_hl.add(Activation('softmax'))

# Construct fully connected NN architecture with 4 hidden layer
model_4_hl = Sequential()
model_4_hl.add(Flatten())
model_4_hl.add(Dense(512))
model_4_hl.add(Activation('relu'))
model_4_hl.add(Dropout(0.5))
model_4_hl.add(Dense(512))
model_4_hl.add(Activation('relu'))
model_4_hl.add(Dropout(0.5))
model_4_hl.add(Dense(512))
model_4_hl.add(Activation('relu'))
model_4_hl.add(Dropout(0.5))
model_4_hl.add(Dense(512))
model_4_hl.add(Activation('relu'))
model_4_hl.add(Dropout(0.5))
model_4_hl.add(Dense(num_classes))
model_4_hl.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model_cnn.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
model_0_hl.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

model_1_hl.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

model_2_hl.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

model_3_hl.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

model_4_hl.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print('Not using data augmentation.')
history_cnn = model_cnn.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)
histories.append(history_cnn)
history_0_hl = model_0_hl.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)
histories.append(history_0_hl)
history_1_hl = model_1_hl.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              shuffle=True)
histories.append(history_1_hl)
history_2_hl = model_2_hl.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              shuffle=True)
histories.append(history_2_hl)
history_3_hl = model_3_hl.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              shuffle=True)
histories.append(history_3_hl)
history_4_hl = model_4_hl.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              shuffle=True)
histories.append(history_4_hl)

for history in histories:
    plt.plot(history.history['acc'])
plt.title('Training Accuracy', fontsize=18)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['cnn', 'nn_0_hl', 'nn_1_hl', 'nn_2_hl', 'nn_3_hl', 'nn_4_hl'], loc='upper left', fontsize=14)
plt.show()

for history in histories:
    plt.plot(history.history['val_acc'])
plt.title('Testing Accuracy', fontsize=18)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['cnn', 'nn_0_hl', 'nn_1_hl', 'nn_2_hl', 'nn_3_hl', 'nn_4_hl'], loc='upper left', fontsize=14)
plt.show()