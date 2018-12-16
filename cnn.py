# coding:utf-8

import os,sys, argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam
from keras.utils import plot_model 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import model


def main(args, classes):
    """ select model"""

    """ log params  """
    para_str = 'CNNmodel_Epoch{}_imgsize{}_Batchsize{}_Adam'.format(
        args.epochs, args.imgsize, args.batchsize)

    """ define callback """
    if not os.path.exists('./tb_log/'):
        os.makedirs('./tb_log/')
    if not os.path.exists('./model_images/'):
        os.makedirs('./model_images/')
    if not os.path.exists('./csv_log/'):
        os.makedirs('./csv_log/')

    tb_cb = TensorBoard(log_dir='/tb_log_'+ para_str,
                        histogram_freq=0,
                        write_grads=False,
                        write_images=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0.5, verbose=1, min_lr=1e-9)
    csv_logger = CSVLogger('./csv_log/' + para_str + '.csv', separator=',')

    """ load image using image data generator """
    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

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

    test_generator = test_datagen.flow_from_directory(
        directory=args.testpath,
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
                    optimizer=Adam(amsgrad=True),
                    metrics=['accuracy'])

    """ train model """
    history = cnn_model.fit_generator(
        generator=train_generator,
        steps_per_epoch = 120//32,
        nb_epoch = args.epochs,
        validation_data = valid_generator,
        nb_val_samples=15)

    """ evaluate model """
    score = cnn_model.evaluate_generator(test_generator)
    print(score)

if __name__ == "__main__":

    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    print ("classes: ", len(classes))

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../dataset/train/')
    parser.add_argument('--validpath', type=str, default='../dataset/valid/')
    parser.add_argument('--testpath', type=str, default='../dataset/test/')
    parser.add_argument('--epochs', '-e', default=50)
    parser.add_argument('--imgsize', '-s', default=256)
    parser.add_argument('--batchsize', '-b', default=32)
    # parser.add_argument('--mode', '-m', )

    args = parser.parse_args()


    main(args, classes)