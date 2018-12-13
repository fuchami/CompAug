# coding:utf-8
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2, Dense
from keras.layers import Activation, Flatten, BatchNormalization, Dropout

# tiny cnn like VGG
def tinycnn_model(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(MaxPooling2(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(MaxPooling2(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(MaxPooling2(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()
    return model