# coding:utf-8

import os,sys, argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam,SGD
from keras.utils import plot_model 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import model

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


def main(args, classes):
    """ select model"""

    """ log params  """
    para_str = 'CNNmodel_Epoch{}_imgsize{}_Batchsize{}_SGD'.format(
        args.epochs, args.imgsize, args.batchsize)

    """ define callback """
    if not os.path.exists('./model_images/'):
        os.makedirs('./model_images/')
    if not os.path.exists('./csv_log/'):
        os.makedirs('./csv_log/')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0.5, verbose=1, min_lr=1e-9)
    csv_logger = CSVLogger('./csv_log/' + para_str + '.csv', separator=',')

    """ load image using image data generator """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

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
        class_mode='categorical',
        shuffle=True)

    """ build cnn model """
    input_shape = (args.imgsize, args.imgsize, 3)
    cnn_model = model.tinycnn_model(input_shape, len(classes))
    plot_model(cnn_model, to_file='./model_images/tinycnn.png', show_shapes=True)

    cnn_model.compile(loss='categorical_crossentropy',
                    optimizer=SGD(),
                    metrics=['accuracy'])

    """ train model """
    history = cnn_model.fit_generator(
        generator=train_generator,
        steps_per_epoch = 120//args.batchsize,
        nb_epoch = args.epochs,
        callbacks=[csv_logger, reduce_lr],
        validation_data = valid_generator,
        validation_steps =15)

    # 学習履歴をプロット
    plot_history(history)

    """ evaluate model """
    # score = cnn_model.evaluate_generator(test_generator)
    # print(score)

if __name__ == "__main__":

    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    print ("classes: ", len(classes))

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../DATASETS/compare_dataset/train_full/')
    parser.add_argument('--validpath', type=str, default='../DATASETS/compare_dataset/valid/')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--imgsize', '-s', type=int, default=256)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    # parser.add_argument('--mode', '-m', )

    args = parser.parse_args()


    main(args, classes)