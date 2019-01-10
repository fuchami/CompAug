# coding:utf-8
""""
画像データをtrain/validation/testに振り分ける前処理

こんな感じにしたい
data
├── train
│   ├── cats   cat0001.jpg - cat1000.jpg
│   └── dogs   dog0001.jpg - dog1000.jpg
├── test
│   ├── cats   cat0001.jpg - cat1000.jpg
│   └── dogs   dog0001.jpg - dog1000.jpg
└── validation
    ├── cats   cat0001.jpg - cat0400.jpg
    └── dogs   dog0001.jpg - dog0400.jpg

"""

import os,sys,argparse
import glob
import random
import shutil
import numpy as np

def split_test_train_valid(args):
    print("datasets split train/test/valid")

    class_path_list = os.listdir(args.srcpath)
    print(class_path_list)

    for class_path in class_path_list:
        print(class_path)

        # クラス内の画像ファイルリストを取得
        img_list = os.listdir(args.srcpath + class_path)
        
        # 取得したリストをシャッフル
        random.shuffle(img_list)

        # 移動先のディレクトリがなければ作成
        if not os.path.exists(args.tarpath + 'train_full/' + class_path):
            os.makedirs(args.tarpath + 'train_full/' + class_path)
        if not os.path.exists(args.tarpath + 'train_half/' + class_path):
            os.makedirs(args.tarpath + 'train_half/' + class_path)
        if not os.path.exists(args.tarpath + 'train_tiny/' + class_path):
            os.makedirs(args.tarpath + 'train_tiny/' + class_path)
        if not os.path.exists(args.tarpath + 'valid/' + class_path):
            os.makedirs(args.tarpath + 'valid/' + class_path)

        # ひとまず全てを訓練データにコピー
        for i in img_list:
            print(i)
            shutil.copyfile("%s%s/%s" % (args.srcpath, class_path, i),
                            "%strain_full/%s/%s" % (args.tarpath, class_path, i))

        img_num = len(img_list)//10
        print("img_num: ", img_num)
        choice = np.random.choice(img_list, img_num * 2 , replace=False)

        # 20%を検証データに
        for i in choice[:img_num*2]:
            shutil.move("%strain_full/%s/%s" % (args.tarpath, class_path, i),
                        "%svalid/%s/%s" % (args.tarpath, class_path, i))
        """
        for i in choice[img_num:]:
            shutil.move("%strain/%s/%s" % (args.tarpath, class_path, i),
                        "%stest/%s/%s" % (args.tarpath, class_path, i))
        """

        # ハーフサイズの訓練データの作成
        img_list = os.listdir(args.tarpath+ 'train_full/' + class_path)
        random.shuffle(img_list)
        choice = np.random.choice(img_list, len(img_list)//2 , replace=False)
        for i in choice[:len(img_list)//2]:
            shutil.copyfile("%strain_full/%s/%s" % (args.tarpath, class_path, i),
                            "%strain_half/%s/%s" % (args.tarpath, class_path, i))
        
        # もっと小さい訓練データの作成
        random.shuffle(img_list)
        choice = np.random.choice(img_list, len(img_list)//2 , replace=False)
        for i in choice[:30]:
            shutil.copyfile("%strain_full/%s/%s" % (args.tarpath, class_path, i),
                            "%strain_tiny/%s/%s" % (args.tarpath, class_path, i))

    return

def split_OneImageChoise(args):
    print("datasets split oneimagetrain/valid")

    class_path_list = os.listdir(args.srcpath)
    print(class_path_list)

    for class_path in class_path_list:
        print(class_path)

        # クラス内の画像ファイルリストを取得
        img_list = os.listdir(args.srcpath + class_path)
        
        # 取得したリストをシャッフル
        random.shuffle(img_list)

        # 移動先のディレクトリがなければ作成
        if not os.path.exists(args.tarpath + 'train/' + class_path):
            os.makedirs(args.tarpath + 'train/' + class_path)
        if not os.path.exists(args.tarpath + 'valid/' + class_path):
            os.makedirs(args.tarpath + 'valid/' + class_path)

        # ひとまず全てを検証データにコピー
        for i in img_list:
            print(i)
            shutil.copyfile("%s%s/%s" % (args.srcpath, class_path, i),
                            "%svalid/%s/%s" % (args.tarpath, class_path, i))

        img_num = 1 
        print("img_num: ", img_num)
        choice = np.random.choice(img_list, img_num, replace=False)

        # 1枚だけ訓練データに
        for i in choice[:img_num]:
            shutil.move("%svalid/%s/%s" % (args.tarpath, class_path, i),
                        "%strain/%s/%s" % (args.tarpath, class_path, i))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='datasets split train/test/valid')
    parser.add_argument('--srcpath', '-s', type=str, default='../DATASETS/datasets_crop/')
    parser.add_argument('--tarpath', '-t', type=str, default='../DATASETS/OneImagetrain/')

    args = parser.parse_args()
    # split_test_train_valid(args)
    split_OneImageChoise(args)
    
