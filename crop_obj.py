# coding:utf-8
"""
データの前処理として物体領域を切り出す
画像1枚に対して1つの物体を切り出す想定

Reference:機械学習のためのOpenCV入門
https://qiita.com/icoxfog417/items/53e61496ad980c41a08e

"""

import os, sys, argparse
import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# matplotlibのフォーマット変換
def to_plt_format(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def binary_threshold(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    under_threshold = 220
    upper_threshold = 30 
    maxvalue = 255

    th, drop_back = cv2.threshold(grayed, under_threshold, maxvalue, cv2.THRESH_BINARY)
    #th, clarify_born = cv2.threshold(grayed, upper_threshold, maxvalue, cv2.THRESH_BINARY_INV)
    #merged = np.minimum(drop_back, clarify_born)

    # plt.imshow(drop_back, 'gray')
    # plt.show()

    return drop_back

# 広めに取る
def paddig_position(x, y, w, h, p):
    return x-p, y-p, w + p*2, h + p*2

def resize_image(img, size):
    img_size = img.shape[:2]
    # if img_size[0] > size[1] or img_size[1] > size[0]:
        # raise Exception("img is larger than size")

    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size)+ [img.shape[2]], dtype=np.uint8)
    resized[row:(row+img.shape[0]), col:(col+img.shape[1])] = img

    mask = np.full(size, 255, dtype=np.uint8)
    mask[row:(row + img.shape[0]), col:(col+img.shape[1])] = 0
    filled = cv2.inpaint(resized, mask, 3, cv2.INPAINT_TELEA)

    return filled

def detect_contour(path):
    cnt=0
    contoured = cv2.imread(path)

    # make binary image
    obj = binary_threshold(path)
    obj = cv2.bitwise_not(obj)

    # detect contour
    im2, contours, hierarchy = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    # draw contour
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        
        # rectangle area
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = paddig_position(x, y, w, h, 60)
        print("はい")
        print(x,y,w,h)

        dst = contoured[y:(y+h), x:(x+w)]
        dst = resize_image(dst, (1024,1024))
        
        save_path = str(0) + '_' + os.path.basename(path)
        cv2.imwrite(save_path, dst)
        cnt+=1

    print(cnt)

def main(args):
    background_list = []

    # ディレクトリ下のファイルをすべてリスト化
    total_image_path_list = glob.glob(args.srcpath + '*')

    # 保存先のディレクトリを作成
    if not os.path.exists(args.tarpath):
        os.makedirs(args.tarpath)

    # 低露出画像は除去
    image_path_list = [x for x in total_image_path_list if not args.lowexposure in x]
    print("remove low exposure images")

    # 背景画像を除去
    for i in image_path_list:
        if 'back' in i:
            background_list.append(i) # 背景画像用のリストへ
            image_path_list.remove(i) # いらない子なのではじく
    print("remove background image")

    print("images:", len(image_path_list))
    print("back ground image", len(background_list))

    for image_path in image_path_list:
        print(image_path)
        detect_contour(image_path)

    """
    for class_path in class_path_list:
        print(class_path)
        img_path_list = os.listdir(src_path+class_path)

        # 保存先ディレクトリを作成
        if not os.path.exists(tar_path + class_path):
            os.makedirs(tar_path + class_path)

        # クロッピング
        for img_path in img_path_list:
            class_src_path = src_path + class_path + '/' + img_path
            class_tar_path = tar_path + class_path + '/'

            cnt = detect_contour(class_path, class_src_path, class_tar_path, min_size, cnt)

    #contoured, crops = detect_contour(path, min_size)
    # contoured = to_plt_format(contoured)

    im_list = np.asarray(contoured)
    plt.imshow(im_list)
    plt.show()
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diff from background and cropping object")

    parser.add_argument('--srcpath', '-s', type=str, default='../datasets/bisco/')
    parser.add_argument('--tarpath', '-t', type=str, default='../datasets_crop/toothbrush/')
    parser.add_argument('--lowexposure', '-l', type=str, default='ex500')

    args = parser.parse_args()

    main(args)