# coding:utf-8
import os,sys
import subprocess

def main():
    cnn_model = ['tiny', 'full', 'v3']
    train_data = ['full', 'half', 'tiny']
    aug_model = ['non', 'aug']
    optimizers = ['SGD', 'Adam', 'AMSGrad']

    for m in cnn_model:
        for t in train_data:
            for a in aug_model:
                for o in optimizers:
                    cmd = ["python3",'cnn.py', "-m", m , "-t", t, "-a", a, "-o", o]
                    print(cmd)
                    result = subprocess.check_output(cmd)
                    print(result)

if __name__ == "__main__":
    main()