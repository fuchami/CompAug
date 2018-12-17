# coding:utf-8
"""
学習の推移を可視化する関数たち

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

def plot_history(history, parastr):

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.xlabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig('accuracy.png')
    plt.close()

    # lossの履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.xlabel('accuracy')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig('accuracy.png')
    plt.close()

 # 混同行列のヒートマップをプロット
def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cxm_data, index=labels, columns=labels)

    plt.figure(figsize = (10, 7))
    sn.heatmap(df_cmx, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("predict classes")
    plt.ylabel("true classes")
    plt.savefig("c_matrix.png")
    plt.close()
    