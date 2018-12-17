# coding:utf-8

import os,sys, argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam,SGD
from keras.utils import plot_model 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

import model
import tools
import load

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
    if not os.path.exists('./train_log/' + para_str + '/'):
        os.makedirs('./train_log/' + para_str + '/')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0.5, verbose=1, min_lr=1e-9)
    csv_logger = CSVLogger('./csv_log/' + para_str + '.csv', separator=',')


    """ load image using image data generator """
    train_generator, valid_generator = load.nonAugmentGenerator(args, classes)

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
        steps_per_epoch = 600//args.batchsize,
        nb_epoch = args.epochs,
        callbacks=[csv_logger],
        validation_data = valid_generator,
        validation_steps =15)

    # 学習履歴をプロット
    tools.plot_history(history, para_str)

    # 混同行列をプロット
    valid_generator.reset()
    Y_pred = cnn_model.predict_generator(valid_generator, steps=150)
    y_pred = np.argmax(Y_pred, axis=1)

    true_classes = valid_generator.classes
    class_label = list(valid_generator.class_indices.keys())

    print('confusion matrix')
    print(confusion_matrix(true_classes, y_pred))
    # tools.confusion_matrix(valid_generator.classes, y_pred, para_str, classes)
    print(classification_report(valid_generator.classes, y_pred, target_names=class_label))


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