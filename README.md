# Compare Data Augmentation

少量のデータでなんとか学習を進める

1. 学習モデルの比較
2. 最適化関数の比較
3. DataAugmentationの比較

# DataAugmentation
### simple data augmentation

### mixup

### Random Erasing

### all mixed augmentation

# Usage

## crop_obj.py
画像から物体の切り出しを行う
  ```
  usage: crop_obj.py [-h] [--srcpath SRCPATH] [--tarpath TARPATH]
                   [--lowexposure LOWEXPOSURE]

diff from background and cropping object

optional arguments:
  -h, --help            show this help message and exit
  --srcpath SRCPATH, -s SRCPATH
  --tarpath TARPATH, -t TARPATH
  --lowexposure LOWEXPOSURE, -l LOWEXPOSURE
  ```

# diff_back.py
背景差分による物体の切り出し
```
usage: diff_back.py [-h] [--srcpath SRCPATH] [--tarpath TARPATH]
                    [--exposure EXPOSURE] [--threshold THRESHOLD]

diff from background and cropping object

optional arguments:
  -h, --help            show this help message and exit
  --srcpath SRCPATH, -s SRCPATH
  --tarpath TARPATH, -t TARPATH
  --exposure EXPOSURE, -l EXPOSURE
  --threshold THRESHOLD
```

# set.py
学習データを訓練データと検証データに振り分ける
```
usage: set.py [-h] [--srcpath SRCPATH] [--tarpath TARPATH]

datasets split train/test/valid

optional arguments:
  -h, --help            show this help message and exit
  --srcpath SRCPATH, -s SRCPATH
  --tarpath TARPATH, -t TARPATH
```

# show_generator.py
ImageDataGeneratorを使って拡張した画像データを描画する

# cnn.py
CNNの学習を行う
```
usage: cnn.py [-h] [--trainpath TRAINPATH] [--trainsize TRAINSIZE]
              [--validpath VALIDPATH] [--epochs EPOCHS] [--imgsize IMGSIZE]
              [--batchsize BATCHSIZE] [--aug_mode AUG_MODE] [--model MODEL]
              [--opt OPT]

train CNN model for classify

optional arguments:
  -h, --help            show this help message and exit
  --trainpath TRAINPATH
  --trainsize TRAINSIZE, -t TRAINSIZE
  --validpath VALIDPATH
  --epochs EPOCHS, -e EPOCHS
  --imgsize IMGSIZE, -s IMGSIZE
  --batchsize BATCHSIZE, -b BATCHSIZE
  --aug_mode AUG_MODE, -a AUG_MODE
                        non, aug, mixup, erasing, fullaug
  --model MODEL, -m MODEL
                        mlp, tiny, full, v3
  --opt OPT, -o OPT     SGD Adam AMSGrad
```

# Reference