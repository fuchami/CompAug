# coding:utf-8
"""
継承して独自のジェネレーターを作成
https://qiita.com/koshian2/items/909360f50e3dd5922f32
mix-upやRandom Erasingを行う

"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# ジェネレーター
def AugmentGenerator(args, classes):
    if args.trainsize == 'full':
        trainpath = args.trainpath + 'train_full/'
    elif args.trainsize == 'half':
        trainpath = args.trainpath + 'train_half/'
    elif args.trainsize == 'tiny':
        trainpath = args.trainpath + 'train_tiny/'
    else:
        raise SyntaxError("please select optimizer: 'full' or 'half' or 'tiny'. ")

    if args.aug_mode == 'non':
        print('-- load image data generator with non augmentation --')
        train_datagen = ImageDataGenerator(rescale=1.0/255)
    elif args.aug_mode == 'aug':
        print('-- load image data generator with augmentation --')
        train_datagen = ImageDataGenerator(rescale=1.0/255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1)
    elif args.aug_mode == 'mixup':
        print('-- load image data generator with mixup --')
        train_datagen = MyImageDataGenerator(rescale=1.0/255,
                                            mix_up_alpha=0.2)
    elif args.aug_mode == 'erasing':
        print('-- load image data generator with random erasing --')
        train_datagen = MyImageDataGenerator(rescale=1.0/255,
                                            random_erasing=True)
    elif args.aug_mode == 'fullaug':
        print('-- load image data generator with FULL AUGMENTATION ! --')
        train_datagen = MyImageDataGenerator(rescale=1.0/255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            mix_up_alpha=0.2,
                                            random_erasing=True)
    else:
        raise SyntaxError("please select ImageDataGenerator: 'non' or 'aug' or 'mixup' or 'erasing' or 'full'. ")

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        trainpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=args.batchsize,
        shuffle=True)

    valid_generator = valid_datagen.flow_from_directory(
        directory=args.validpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        shuffle=True)

    return train_generator, valid_generator

class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, featurewise_center=False, samplewise_center=False,
                featurewise_std_normalization=False, sampleWise_std_normalization=False,
                zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
                channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
                vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, 
                validation_split=0.0, random_crop=None, mix_up_alpha=0.0, random_erasing=False):

        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization,
                        sampleWise_std_normalization, zca_whitening, zca_epsilon, rotation_range,
                        width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range,
                        channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale,
                        preprocessing_function, data_format, validation_split)
        # 拡張処理のパラメータ
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        assert random_crop == None or len(random_crop) == 2
        self.random_crop_size = random_crop

        self.random_erasing = random_erasing

    """ Random Crop """
    def random_crop(self, original_img):
        # Note: image_data_format is 'channel_last'
        assert original_img.shape[2] == 3
        if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
            raise ValueError(" はい" )
        
        height, width = original_img.shape[0], original_img.shape[1]
        dy, dx = self.random_crop_size
        x = np.random.randint(0, width -dx + 1)
        y = np.random.randint(0, height -dy + 1)
        return original_img[y:(y+dy), x:(x+dx), :]

    """ Mix up """
    def mix_up(self,x1, y1, x2, y2):
        # assert x1.shape[0] == y1.shape ==[0] == x2.shape[0] == y2.shape[0]
        batch_size = x1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        x_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = x1 * x_l + x2 * (1- x_l)
        Y = y1 * y_l + y2 * (1- y_l)
        return X, Y
    
    """ Random Erasing 
    https://www.kumilog.net/entry/numpy-data-augmentation
    """
    def random_eraser(self, original_img):
        image = np.copy(original_img)
        p=0.5
        s=(0.02, 0.4)
        r=(0.3, 3)

        # マスクするかしないか
        if np.random.rand() > p:
            return image

        # マスクする画素値をランダムで決める
        mask_value = np.random.random()

        h, w, _ = image.shape
        # マスクサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
        mask_area = np.random.randint(h * w * s[0], h* w * s[1])

        # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]
        
        # マスクのサイズとアスペクト比からマスクの高さと幅を決める
        # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > h-1:
            mask_height = h-1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w-1:
            mask_width = w-1
        
        top = np.random.randint(0, h-mask_height)
        left = np.random.randint(0, w-mask_width)
        bottom = top+mask_height
        right = left+mask_width

        image[top:bottom, left:right, :].fill(mask_value)

        return image
    
    """ バッチサイズごとに画像を読み込んで返す """
    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=None, save_prefix='', save_format='png', 
                            follow_links=False, subset=None, interpolation='nearest'):
        # 親クラスのflow_from_directory
        batches = super().flow_from_directory(directory, target_size, color_mode, classes, class_mode,
                                            batch_size, shuffle, seed, save_to_dir, save_prefix, 
                                            save_format, follow_links, subset, interpolation)
        # 拡張処理
        while True:
            batch_x, batch_y = next(batches)
            """ random crop """
            if self.random_crop_size != None:
                x = np.zeros((batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_crop(batch_x[i])
                batch_x = x

            """ mix up """
            if self.mix_up_alpha > 0:
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)

            """ random erasing """
            if self.random_erasing == True:
                x = np.zeros((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 3))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_eraser(batch_x[i])
                batch_x = x
            
            yield(batch_x, batch_y)
