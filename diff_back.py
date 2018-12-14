# coding:utf-8
"""
背景差分を用いて物体領域短形を切り取る

"""

import os,sys,glob,argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# matplotlibのフォーマット形式に
def to_plt_format(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# 広めに取る
def paddig_position(x, y, w, h, p):
    return x-p, y-p, w + p*2, h + p*2

#  輪郭抽出
def getContours(img_path, diff_img, tarpath, areas_max):
    print("get contours:", img_path)
    c = 0
    img = cv2.imread(img_path, 1)
    contours = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    result_list = []

    #print(len(contours))
    area_list =[]

    for cnt in contours:
        
        # 面積をリストへ
        area = cv2.contourArea(cnt)
        area_list.append(area)

    # 面積最大のやつをひっこぬく
    for i in range(areas_max):

        max_idx = area_list.index(max(area_list))
        x, y, w, h = cv2.boundingRect(contours[max_idx])

        contours.pop(max_idx)
        area_list.pop(max_idx)

        x, y, w, h = paddig_position(x, y, w, h, 100)

        # 短形描画
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

        # 切り取り保存        
        basename = os.path.basename(img_path)
        name, ext = os.path.splitext(basename)
        save_path = tarpath + name + '_' + str(i) + ext
        print("save img:", save_path)

        dst = img[y:(y+h), x:(x+w)]
        cv2.imwrite(save_path, dst)
        c+=1

        # 画像確認用
        # result = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.imshow(result)
        # plt.show()

    return 

# 背景差分
def getdiff(img, bgimg):
    """ back ground img """
    # img read gray scale 
    bgimg = cv2.imread(bgimg, 0)
    # gaussian blur
    bgimg = cv2.GaussianBlur(bgimg, (11,11), 0)
    # binary img 
    bgimg = cv2.threshold(bgimg, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    """ object img """
    img = cv2.imread(img, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    """ get diff """
    diff_img = cv2.absdiff(bgimg, img_gray)
    diff_img = cv2.threshold(diff_img, 180, 255, cv2.THRESH_BINARY)[1]

    operator = np.ones((3,3), np.uint8)
    img_dilate = cv2.dilate(diff_img, operator, iterations=4)
    img_mask = cv2.erode(img_dilate, operator, iterations=4)
    img_dst = cv2.bitwise_and(img_gray, img_mask)
    cv2.imwrite('mask.png', img_dst)

    return img_dst

def main(args):

    # ディレクトリ下のファイルをすべてリスト化
    total_image_path_list = glob.glob(args.srcpath + '*')

    # 保存先ディレクトリを作成
    if not os.path.exists(args.tarpath):
        os.makedirs(args.tarpath)

    # 低露出画像は除去
    print("remove low exposure images")
    image_path_list = [x for x in total_image_path_list if args.exposure not in x]

    # 背景画像を除去
    print("remove background image")
    background_list = [p for p in image_path_list if 'background' in p]
    image_path_list = [p for p in image_path_list if 'background' not in p]

    print("images:", len(image_path_list) )
    print("back ground image", len(background_list), background_list)
    
    # 背景画像
    for image_path in image_path_list:

        # 背景画像を決定
        num = image_path.split('_')
        print(num[1]) # => 22711165
        bgimg = [x for x in background_list if num[1] in x]
        print(bgimg) # => ['../datasets/bisco/camNo_22737280_00005485_ex24000_background.png']

        # 対応するものがなければ終了
        if not bgimg:
            print("not found background image !!!: ", num[1])
            sys.exit()

        # 背景差分
        diff_img = getdiff(image_path, bgimg[0])
        # 輪郭抽出
        getContours(image_path, diff_img, args.tarpath, args.threshold)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diff from background and cropping object")

    parser.add_argument('--srcpath', '-s', type=str, default='../datasets/clearclean/')
    parser.add_argument('--tarpath', '-t', type=str, default='../datasets_crop/clearclean/')
    parser.add_argument('--exposure', '-l', type=str, default='ex500')
    parser.add_argument('--threshold', type=int, default=2)

    args = parser.parse_args()
    main(args)