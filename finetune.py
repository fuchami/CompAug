# coding:utf-8
"""
MLPで物体認識
"""

import os,sys,argparse,csv
import numpy as np
import h5py

import keras
from keras.optimizers import Adam,SGD,rmsprop
from keras.utils import plot_model
from keras.callbacks import CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

import tools, mlp_load, model, load

def main(args, classes):

    """ log params """
    para_str = 'densenet_augmode_{}{}_epoch{}_imgsize{}_batchsize{}_{}{}'.format(
        args.aug_mode, args.aug_para, args.epochs, args.imgsize, args.batchsize, args.opt,args.lr)
    print("start this params train: ", para_str)
    para_path = './densenet_log/' + para_str

    """ define callback """
    if not os.path.exists('./images/'):
        os.makedirs('./images/')
    if not os.path.exists( para_path + '/'):
        os.makedirs( para_path + '/')
    if not os.path.exists('./densenet_log/log.csv'):
        with open('./densenet_log/log.csv', 'w')as f:
            writer = csv.writer(f)
            header = ['traindata_size', 'augmentation_mode', 'optimizer', 'validation accuracy', 'validation loss']
            writer.writerow(header)
    
    base_lr = args.lr 
    lr_decay_rate = 1/10
    lr_steps = 4
    reduce_lr = LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // args.epochs)), verbose=1)
    csv_logger = CSVLogger( para_path + '/log.csv', separator=',')
    callbacks = []
    callbacks.append(csv_logger)
    callbacks.append(reduce_lr)

    """ load image using image data generator """
    train_generator, valid_generator = load.OneImageGenerator(args, classes)

    """ build model """
    input_shape = (args.imgsize, args.imgsize, 3, )
    densenet_model = model.densenet_finetue(input_shape, len(classes))

    """ select optimizer """
    if args.opt == 'SGD':
        opt = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
        print("-- optimizer: SGD --")
    else:
        raise SyntaxError("please select optimizer: 'SGD' or 'Adam'. ")
    
    plot_model(densenet_model, to_file='./images/model.densenet_finetue.png', show_shapes=True)

    densenet_model.compile(loss='categorical_crossentropy',
                    optimizer= opt,
                    metrics=['accuracy'])

    """ train model """
    history = densenet_model.fit_generator(
        generator=train_generator,
        steps_per_epoch = 5,
        nb_epoch = args.epochs,
        callbacks = callbacks,
        validation_data = valid_generator,
        validation_steps = 149// args.batchsize)
    
    # 学習履歴をプロット
    tools.plot_history(history, para_str, para_path)

    """ evaluate model """
    valid_generator.reset()
    score = densenet_model.evaluate_generator(generator=valid_generator, steps=valid_generator.samples)
    print("model score:", score)

    """ 学習結果をCSV出力 """
    with open('./densenet_log/log.csv', 'a') as f:
        data = [args.trainsize, args.aug_mode, args.opt, score[1], score[0]]
        writer = csv.writer(f)
        writer.writerow(data)

if __name__ == "__main__":

    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    print ("classes: ", len(classes))

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../DATASETS/OneImagetrain/train/')
    parser.add_argument('--trainsize', '-t', type=str, default='full')
    parser.add_argument('--validpath', type=str, default='../DATASETS/OneImagetrain/valid/')
    parser.add_argument('--epochs', '-e', type=int, default=80)
    parser.add_argument('--imgsize', '-s', type=int, default=224)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    # 水増しなし 水増しあり mixup を選択
    parser.add_argument('--aug_mode', '-a', default='non',
                        help='non, aug, mixup, erasing, fullaug')
    parser.add_argument('--aug_para', '-p', type=float, default=0.1)
    # 最適化関数
    parser.add_argument('--opt', '-o', default='SGD',
                        help='SGD Adam AMSGrad ')
    parser.add_argument('--lr', '-l', type=float, default=0.01)

    args = parser.parse_args()

    main(args, classes)