# coding:utf-8

import os,sys, argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam,SGD
from keras.utils import plot_model 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report

import model, load, tools

def main(args, classes):

    """ log params  """
    para_str = 'model_{}_trainsize_{}_{}_Epoch{}_imgsize{}_Batchsize{}_{}'.format(
        args.model, args.trainsize, args.aug_mode, args.epochs, args.imgsize, args.batchsize, args.opt)
    print("start this params CNN train: ", para_str)

    """ define callback """
    if not os.path.exists('./model_images/'):
        os.makedirs('./model_images/')
    if not os.path.exists('./train_log/' + para_str + '/'):
        os.makedirs('./train_log/' + para_str + '/')

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-9)
    base_lr = 1e-3  # adamとかならこのくらい。SGDなら例えば 0.1 * batch_size / 128 とかくらい。
    lr_decay_rate = 1 / 3
    lr_steps = 4
    reduce_lr = LearningRateScheduler(lambda ep: float(base_lr * lr_decay_rate ** (ep * lr_steps // args.epochs)))
    es_cb = EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1, mode='auto')
    csv_logger = CSVLogger('./train_log/' + para_str + '/' + 'log.csv', separator=',')

    callbacks = []
    callbacks.append(csv_logger)
    callbacks.append(reduce_lr)

    """ load image using image data generator """
    train_generator, valid_generator = load.AugmentGenerator(args, classes)

    print("train generator samples: ", train_generator.samples)
    print("valid generator samples: ", valid_generator.samples)
    # print(train_generator.class_indices)
    
    """ build cnn model """
    input_shape = (args.imgsize, args.imgsize, 3)
    # cnn_model = model.tinycnn_model(input_shape, len(classes))
    """ select load model """
    if args.model == 'tiny':
        cnn_model = model.tinycnn_model(input_shape, len(classes))
    if args.model == 'mlp':
        cnn_model = model.mlp(input_shape, len(classes))
    elif args.model == 'full':
        cnn_model = model.cnn_fullmodel(input_shape, len(classes))
    elif args.model == 'v3':
        cnn_model = model.inceptionv3_finetune_model(input_shape, len(classes))
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

    plot_model(cnn_model, to_file='./model_images/' +str(args.model)+  'model.png', show_shapes=True)

    cnn_model.compile(loss='categorical_crossentropy',
                    optimizer= opt,
                    metrics=['accuracy'])

    """ train model """
    history = cnn_model.fit_generator(
        generator=train_generator,
        steps_per_epoch = train_generator.samples// train_generator.batch_size,
        nb_epoch = args.epochs,
        callbacks = callbacks,
        validation_data = valid_generator,
        validation_steps = valid_generator.samples)
    
    # 学習履歴をプロット
    tools.plot_history(history, para_str)

    """ evaluate model """
    valid_generator.reset()
    score =cnn_model.evaluate_generator(generator=valid_generator, steps=valid_generator.samples)
    print("model score: ",score)

    """ 学習結果をテキスト出力 """
    with open('./train_log/log.txt', 'a') as f:
        f.write('-------------------------------------------------------\n')
        f.write(para_str+'\n')
        acc = 'validation acc: ' + str(score[1])
        f.write(acc+'\n')
        f.write('-------------------------------------------------------\n')
        f.write('\n\n')

if __name__ == "__main__":

    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    print ("classes: ", len(classes))

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../DATASETS/compare_dataset/')
    parser.add_argument('--trainsize', '-t', type=str, default='full')
    parser.add_argument('--validpath', type=str, default='../DATASETS/compare_dataset/valid/')
    parser.add_argument('--epochs', '-e', type=int, default=80)
    parser.add_argument('--imgsize', '-s', type=int, default=64)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    # 水増しなし 水増しあり mixup を選択
    parser.add_argument('--aug_mode', '-a', default='non',
                        help='non, aug, mixup, erasing, full)
    # 学習させるモデルの選択
    parser.add_argument('--model', '-m', default='tiny',
                        help='mlp, tiny, full, v3')
    # 最適化関数
    parser.add_argument('--opt', '-o', default='SGD',
                        help='SGD Adam AMSGrad ')

    args = parser.parse_args()

    main(args, classes)
