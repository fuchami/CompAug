# coding:utf-8
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, BatchNormalization, Dropout

# 小さな畳み込みニューラルネットワーク
def tinycnn_model(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()
    return model


def inceptionv3_finetune_model(input_shape, classes):

    model = Sequential()

    return model
