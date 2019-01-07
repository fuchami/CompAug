# coding:utf-8
import os,sys
import subprocess

def main():
    cnn_model = ['tiny', 'full']
    train_data = ['full', 'half', 'tiny']
    aug_model = ['non', 'aug', 'mixup', 'erasing', 'fullaug']
    optimizers = ['SGD', 'Adam', 'AMSGrad']

    for o in optimizers:
        for m in cnn_model:
            for t in train_data:
                for a in aug_model:
                    cmd = ["python",'cnn.py', "-m", m , "-t", t, "-a", a, "-o", o ]
                    print(cmd)
                    result = subprocess.check_output(cmd)
                    print(result)

if __name__ == "__main__":
    main()
