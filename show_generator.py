# coding:utf-8
"""
ImageDataGeneratorで生成された画像を可視化

"""

import numpy as np
import matplotlib.pyplot as plt
import load

def show_img(imgs, row, col):
    if len(imgs) != (row*col):
        raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))
    
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, img in enumerate(imgs):
        plot_num = i+1
        ax = fig.add_subplot(row, col, plot_num, xticks=[], yticks=[])
        ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    classes = ['bisco','clearclean', 'frisk', 'toothbrush', 'udon']
    """
    datagen = load.MyImageDataGenerator(
                        rescale=1/255.0,
                        mix_up_alpha=4)
    """
    datagen = load.MyImageDataGenerator(rescale=1/255.0,
                                        shear_range=0.2,
                                        zoom_range=0.5,
                                        width_shift_range=0.3,
                                        height_shift_range=0.2,
                                        mix_up_alpha=0.2,
                                        random_erasing=True)

    
    max_img_num = 12
    imgs = []
    for d in datagen.flow_from_directory('/Users/yuuki/Downloads/all/DogsCats/',
                                            batch_size=1, target_size=(128,128),
                                            class_mode='categorical',
                                            ):
        imgs.append(np.squeeze(d[0], axis=0))
        if (len(imgs) % max_img_num) == 0:
            break
    
    show_img(imgs, row=4, col=3)