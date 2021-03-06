# coding:utf-8
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, BatchNormalization, Dropout, Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121

# multi layer perceptron
def mlp(input_shape, classes):
    inputs = Input(input_shape,)

    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.summary()
    return model

# 小さな畳み込みニューラルネットワーク
def tinycnn_model(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(2,2) ))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2) ))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(2,2) ))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

# ガチ構成のCNNモデル
def cnn_fullmodel(input_shape, classes):

    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3,3), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

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

def densenet_finetue(input_shape, classes):
    Input_shape = Input(shape=input_shape)
    densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_tensor=Input_shape)
    densenet.trainable = False
    # densenet.summary()

    x_in = Input_shape
    x = densenet(x_in)
    x = Dense(128, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(x_in, x)

    model.summary()

    return model

