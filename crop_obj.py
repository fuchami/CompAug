# coding:utf-8
"""
データの前処理として物体領域を切り出す

Reference:機械学習のためのOpenCV入門
https://qiita.com/icoxfog417/items/53e61496ad980c41a08e

"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def to_grayscale(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

def to_plt_format(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def binary_threshold(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    under_threshold = 100
    upper_threshold = 230
    maxvalue = 255
    th, drop_back = cv2.threshold(grayed, under_threshold, maxvalue, cv2.THRESH_BINARY)
    th, clarify_born = cv2.threshold(grayed, upper_threshold, maxvalue, cv2.THRESH_BINARY)
    merged = np.minimum(drop_back, clarify_born)

    return merged

def paddig_position(x, y, w, h, p):
    return x-p, y-p, w+p *2, h+p *2

def resize_image(img, size):
    img_size = img.shape[:2]
    if img_size[0] > size[1] or img_size[1] > size[0]:
        raise Exception("img is larger than size")

    row = (size[1] - img_size[0])
    col = (size[0] - img_size[1])
    resized = np.zeros(list(size)+ [img.shape[2]], dtype=np.uint8)
    resized[row:(row+img.shape[0]), col:(col+img.shape[1])] = img

    mask = np.full(size, 255, dtype=np.uint8)
    mask[row:(row + img.shape[0]), col:(col+img.shape[1])] = 0
    filled = cv2.inpaint(resized, mask, 3, cv2.INPAINT_TELEA)

    return filled

def detect_contour(path, min_size):
    contoured = cv2.imread(path)
    forcrop = cv2.imread(path)

    # make binary image
    obj = binary_threshold(path)
    obj = cv2.bitwise_not(obj)

    # detect contour
    im2, contours, hierarchy = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    # draw contour
    for c in contours:
        if cv2.contourArea(c) < min_size:
            continue
        
        # rectangle area
        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = paddig_position(x, y, w, h, 50)

        # crop the image
        cropped = forcrop[y:(y+h), x:(x+w)]
        #cropped = resize_image(cropped, (256,256))
        crops.append(cropped)

        # draw contour
        cv2.drawContours(contoured, c, -1, (0, 0, 255), -3) # contour
        cv2.rectangle(contoured, (x, y), (x+w, y+h), (0, 255, 0), 3) # ectangle
    
    return contoured, crops

def main():
    path = '../SINEPOST/train/toothpaste/camNo_22711133_00005403_20_ex26000.png'
    min_size = 512

    contoured, crops = detect_contour(path, min_size)
    contoured = to_plt_format(contoured)

    im_list = np.asarray(contoured)
    plt.imshow(im_list)
    plt.show()


if __name__ == "__main__":
    main()