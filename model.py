# coding:utf-8
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, BatchNormalization, Dropout, Input
from keras.applications.inception_v3 import InceptionV3

# 小さな畳み込みニューラルネットワーク
def tinycnn_model(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

# ガチ構成のCNNモデル
def cnn_fullmodel(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1028))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

# inceptionV3をfine tuning
def inceptionv3_finetune_model(input_shape, classes):

    Input_shape = Input(shape=input_shape)
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_tensor=Input_shape)
    base_model.trainable = False
    base_model.summary()

    x_in = Input_shape
    x = base_model(x_in)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(x_in, x)

    model.summary()

    return model
