# coding:utf-8
"""
継承して独自のジェネレーターを作成

mix-upやRandom Erasingを行う

"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# 水増し無しのジェネレーター
def nonAugmentGenerator(args, classes):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_dagaten = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        args.trainpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=args.batchsize,
        shuffle=True)

    valid_generator = valid_datagen.flow_from_directory(
        directory=args.validpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical')
    
    return train_generator, valid_generator

# 水増し有りのジェネレーター
def AUgmentGenerator():

    return