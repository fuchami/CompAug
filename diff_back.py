# coding:utf-8
"""
背景差分を用いて物体領域短形を切り取る

"""

import os,sys,glob,argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def diffcrop(path):

    return

def main(args):
    background_list = []

    # ディレクトリ下のファイルをすべてリスト化
    total_image_path_list = glob.glob(args.srcpath + '*')

    # 低露出画像は除去
    image_path_list = [x for x in total_image_path_list if not args.lowexposure in x]
    print("remove low exposure images")

    # 背景画像を除去
    for i in image_path_list:
        if 'back' in i:
            background_list.append(i) # 背景画像用のリストへ
            image_path_list.remove(i) # いらない子なのではじく
    print("remove background image")

    for i in image_path_list:
        print(i)

    print("images:", len(image_path_list))
    print("back ground image", len(background_list))

    # 背景画像
    for image_path in image_path_list:
        print(image_path)
        diffcrop(image_path)



    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diff from background and cropping object")

    parser.add_argument('--srcpath', '-s', type=str, default='../datasets/bisco/')
    parser.add_argument('--tarpath', '-t', type=str, default='')
    parser.add_argument('--lowexposure', '-l', type=str, default='ex500')

    args = parser.parse_args()
    main(args)