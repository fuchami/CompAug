# coding:utf-8

import os,sys, argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam,SGD
from keras.utils import plot_model 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report

import model
import tools
import load

def main(args, classes):

    """ log params  """
    para_str = 'model_{}_Dtype_{}_Epoch{}_imgsize{}_Batchsize{}_SGD'.format(
        args.model, args.aug_mode, args.epochs, args.imgsize, args.batchsize)

    """ define callback """
    if not os.path.exists('./model_images/'):
        os.makedirs('./model_images/')
    if not os.path.exists('./csv_log/'):
        os.makedirs('./csv_log/')
    if not os.path.exists('./train_log/' + para_str + '/'):
        os.makedirs('./train_log/' + para_str + '/')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-9)
    es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
    csv_logger = CSVLogger('./csv_log/' + para_str + '.csv', separator=',')


    """ load image using image data generator """
    if args.aug_mode == 'None':
        print("-- load image generator with non augmentation --")
        train_generator, valid_generator = load.nonAugmentGenerator(args, classes)
    elif args.aug_mode == 'aug':
        print("-- load image generator with augmentation --")
        train_generator, valid_generator = load.AugmentGenerator(args, classes)
    elif args.aug_mode =='mixup':
        print("-- load image generator with mixup --")
    else:
        raise SyntaxError("please select ImageDataGenerator : 'None' or 'aug' or 'mixup'. ")

    print("train generator samples: ", train_generator.samples)
    print("valid generator samples: ", valid_generator.samples)
    print(train_generator.class_indices)
    

    """ build cnn model """
    input_shape = (args.imgsize, args.imgsize, 3)
    # cnn_model = model.tinycnn_model(input_shape, len(classes))
    """ select load model """
    if args.model == 'tiny':
        cnn_model = model.tinycnn_model(input_shape, len(classes))
    elif args.model == 'full':
        cnn_model = model.cnn_fullmodel(input_shape, len(classes))
    elif args.model == 'v3':
        cnn_model = model.inceptionv3_finetune_model(input_shape, len(classes))
    else:
        raise SyntaxError("please select model : 'tiny' or 'full' or 'v3'. ")
    
    """ select optimizer """
    if args.opt == 'SGD':
        opt = SGD(lr=1e-4, momentum=0.9)
        print("-- optimizer: SGD --")
    elif args.opt == 'Adam':
        opt = Adam()
        print("-- optimizer: Adam --")
    elif args.opt == 'AMSGrad':
        opt = Adam(amsgrad=True)
        print("-- optimizer: AMSGrad --")
    else:
        raise SyntaxError("please select optimizer: 'SGD' or 'Adam' or 'AMSGrad'. ")

    plot_model(cnn_model, to_file='./model_images/tinycnn.png', show_shapes=True)

    cnn_model.compile(loss='categorical_crossentropy',
                    optimizer= opt,
                    metrics=['accuracy'])

    """ train model """
    history = cnn_model.fit_generator(
        generator=train_generator,
        steps_per_epoch = train_generator.samples// train_generator.batch_size,
        nb_epoch = args.epochs,
        callbacks=[csv_logger, reduce_lr],
        validation_data = valid_generator,
        validation_steps = valid_generator.samples// args.batchsize)
    
    # 学習履歴をプロット
    tools.plot_history(history, para_str)

    """ evaluate model """
    score =cnn_model.evaluate_generator(generator=valid_generator, steps=valid_generator.samples)
    print("model score: ",score)
    score =cnn_model.evaluate_generator(generator=valid_generator, steps=valid_generator.samples)
    print("model score: ",score)
    score =cnn_model.evaluate_generator(generator=valid_generator, steps=valid_generator.samples)
    print("model score: ",score)

    """ 学習結果をテキスト出力 """
    with open('./train_log/log.txt', 'a') as f:
        f.write('--------------------------------------')
        f.write(para_str)
        f.write('model score: ' + score)
        f.write('--------------------------------------')

    """ confusion matrix 
    valid_generator.reset()
    ground_truth = valid_generator.classes
    print("ground_truth:", ground_truth)
    predictions = cnn_model.predict_generator(valid_generator, verbose=1, steps=valid_generator.samples//30)
    predicted_classes = np.argmax(predictions, axis=1)
    print("predicted_classes: ", predicted_classes)

    cm = confusion_matrix(ground_truth, predicted_classes)
    print(cm)
    """

if __name__ == "__main__":

    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    print ("classes: ", len(classes))

    parser = argparse.ArgumentParser(description='train CNN model for classify')
    parser.add_argument('--trainpath', type=str, default='../DATASETS/compare_dataset/train_full/')
    parser.add_argument('--validpath', type=str, default='../DATASETS/compare_dataset/valid/')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--imgsize', '-s', type=int, default=128)
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    # 水増しなし 水増しあり mixup を選択
    parser.add_argument('--aug_mode', '-a', default='aug',
                        help='None: Non Augmenration, aug: simpleAugmentation, mixup')
    # 学習させるモデルの選択
    parser.add_argument('--model', '-m', default="v3",
                        help='tiny, full, v3')
    # 最適化関数
    parser.add_argument('--opt', '-o', default="Adam",
                        help='SGD Adam AMSGrad ')

    args = parser.parse_args()

    main(args, classes)