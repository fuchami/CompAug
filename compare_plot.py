# coding:utf-8
"""
各比較したやつを読み込んでプロット

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compare_mlp():
    non = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_non_para0.2_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'baseline','loss'], header=0)
    aug = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_aug_para0.1_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'augmentation','loss'], header=0)
    erasing = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_erasing_para0.1_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'random_erasing','loss'], header=0)
    aug_mixup = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_aug_mixup_para0.1_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'augmentation+mixup','loss'], header=0)
    aug_erasing = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_aug_erasing_para0.1_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'augmentation+random_erasing','loss'], header=0)
    aug_mixup_erasing = pd.read_csv('../report/20190115/mlp_log/mlp_trainsize_tiny_aug_mixup_erasing_para0.1_epoch80_imgsize32_batchsize32_optSGD0.01/log.csv', 
                        names=['e','a','l', 'augmentation+mixup+random_erasing','loss'], header=0)

    plt.plot(range(0,80), non['baseline'])
    plt.plot(range(0,80), aug['augmentation'])
    plt.plot(range(0,80), erasing['random_erasing'])
    plt.plot(range(0,80), aug_mixup['augmentation+mixup'])
    plt.plot(range(0,80), aug_erasing['augmentation+random_erasing'])
    plt.plot(range(0,80), aug_mixup_erasing['augmentation+mixup+random_erasing'])
    plt.title('compare augmentation with MLP')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    return

def compare_finetune():
    non = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_non0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', 'baseline','loss'], header=0)
    aug = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_aug0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', '+augmentation','loss'], header=0)
    mixup = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_mixup0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', '+mixup','loss'], header=0)
    erasing = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_erasing0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', '+random_erasing','loss'], header=0)
    aug_mixup = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_aug_mixup0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', '+augmentation+mixup','loss'], header=0)
    aug_erasing = pd.read_csv('../report/20190115/densenet_log/densenet_augmode_aug_erasing0.1_epoch80_imgsize224_batchsize16_SGD0.01/log.csv', 
                        names=['e','a','l', '+augmentation+random_erasing','loss'], header=0)

    plt.plot(range(0,80), non['baseline'])
    plt.plot(range(0,80), aug['+augmentation'])
    plt.plot(range(0,80), mixup['+mixup'])
    plt.plot(range(0,80), erasing['+random_erasing'])
    plt.plot(range(0,80), aug_mixup['+augmentation+mixup'])
    plt.plot(range(0,80), aug_erasing['+augmentation+random_erasing'])
    plt.title('compare augmentation with finetuning-Densenet')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    return
if __name__ == "__main__":

    # compare_mlp()
    compare_finetune()
