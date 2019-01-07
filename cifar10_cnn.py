# coding:utf-8

import os,sys,argparse,csv
import h5py
import numpy as np

import keras
from keras.optimizers import Adam,SGD
from keras.utils import plot_model
from keras.callbacks import TensorBoard,EarlyStopping,ReduceLROnPlateau,CSVLogger,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

import model, load, tools

def main(args):

    """ log params  """
    para_str = 'cifar10model_{}_{}_Epoch{}_batchsize_{}_{}'.format(
        args.model, args.aug_mode, args.epochs, args.batchsize, args.opt)
    print("start this params CNN train: ", para_str)

    """ define callback """
    if not os.path.exists('./model_images/'):
        os.makedirs('./model_images/')
    if not os.path.exists('./cifar10_train_log/' + para_str + '/'):
        os.makedirs('./cifar10_train_log/' + para_str + '/')
    if not os.path.exists('./cifar10_train_log/log.csv'):
        with open('./cifar10_train_log/cifar10_log.csv', 'w')as f:
            writer = csv.writer(f)
            header = ['model', 'traindata_size', 'augmentation_mode', 'optimizer', 'validation accuracy', 'validation loss']
            writer.writerow(header)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, min_lr=1e-9)
    """
    base_lr = 1e-3  # adamとかならこのくらい。SGDなら例えば 0.1 * batch_size / 128 とかくらい。
    lr_decay_rate = 1 / 3
    lr_steps = 4
    reduce_lr = LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // args.epochs)), verbose=1)
    """
    es_cb = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1, mode='auto')
    csv_logger = CSVLogger('./cifar10_train_log/' + para_str + '/' + 'cifar10_log.csv', separator=',')

    callbacks = []
    callbacks.append(csv_logger)
    callbacks.append(reduce_lr)

    """ load cifar10 datasets"""
    print("-- load cifar10 datasets --")
    train_generator, x_train, y_train, x_test, y_test = load.cifar10Generator(args)

    """ build cnn model """
    input_shape = (args.imgsize, args.imgsize, 3)
    """ select load model """
    if args.model == 'tiny':
        cnn_model = model.tinycnn_model(input_shape, 10)
    elif args.model == 'full':
        cnn_model = model.cnn_fullmodel(input_shape, 10)
    elif args.model == 'v3':
        cnn_model = model.inceptionv3_finetune_model(input_shape, 10)
    elif args.model == 'mlp':
        cnn_model = model.mlp(input_shape, 10)
    else:
        raise SyntaxError("please select model : 'tiny' or 'full' or 'v3'. ")
    
    """ select optimizer """
    if args.opt == 'SGD':
        opt = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
        print("-- optimizer: SGD --")
    elif args.opt == 'Adam':
        opt = Adam()
        print("-- optimizer: Adam --")
    elif args.opt == 'AMSGrad':
        opt = Adam(amsgrad=True)
        print("-- optimizer: AMSGrad --")
    else:
        raise SyntaxError("please select optimizer: 'SGD' or 'Adam' or 'AMSGrad'. ")

    cnn_model.compile(loss='categorical_crossentropy',
                    optimizer= opt,
                    metrics=['accuracy'])

    """ train model """
    history = cnn_model.fit_generator(train_generator.flow(x_train, y_train, batch_size=args.batchsize),
                                    steps_per_epoch = x_train.shape[0] // args.batchsize,
                                    validation_data = (x_test, y_test),
                                    epochs = args.epochs,
                                    callbacks = callbacks)

    # 学習履歴をプロット
    tools.plot_history(history, para_str)

    """ evaluate model """
    score =cnn_model.evaluate(x_test, y_test, verbose=1)
    print("model score: ", score)

    """ 学習結果をCSV出力 """
    with open('./cifar10_train_log/cifar10_log.csv', 'a') as f:
        data = [args.model,args.aug_mode, args.opt, score[1], score[0]]
        writer = csv.writer(f)
        writer.writerow(data)
    
    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../DATASETS/compare_dataset/')
    parser.add_argument('--validpath', type=str, default='../DATASETS/compare_dataset/valid/')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--imgsize', '-s', type=int, default=32)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    # 水増しなし 水増しあり mixup を選択
    parser.add_argument('--aug_mode', '-a', default='non',
                        help='non, aug, mixup, erasing, fullaug')
    # 学習させるモデルの選択
    parser.add_argument('--model', '-m', default='tiny',
                        help='mlp, tiny, full, v3')
    # 最適化関数
    parser.add_argument('--opt', '-o', default='Adam',
                        help='SGD Adam AMSGrad ')

    args = parser.parse_args()

    main(args)
