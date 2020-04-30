import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy(tl_label, pre_res):
    label_list_ = [0, 0, 0, 0] 
    count = 0
    count_no3 = 0
    count_no3_all = 0
    for tl, pre in zip(tl_label, pre_res):
        if int(tl) == int(pre):
            count += 1
            if int(tl) == int(pre) and int(pre) != 3:
                count_no3 += 1
        else:
            if tl == 0:
                label_list_[0] += 1
            elif tl == 1:
                label_list_[1] += 1
            elif tl == 2:
                label_list_[2] += 1
            elif tl == 3:
                label_list_[3] += 1

        if tl != 3:
            count_no3_all += 1

    print("正解/総数は", str(count) + "/" + str(len(tl_label)))
    print("精度(unknow含む)は", count/len(tl_label) * 100, "%です。")
    print("精度(unknow除く)は", count_no3/count_no3_all * 100, "%です。")
    print("間違いlabel [red, blue, yellow, unknown] --> ", label_list_)
    eval_confusion_matrix(tl_label, pre_res)

def eval_confusion_matrix(tl_label, pre_res):
    #label=["red", "blue", "yellow", "unknown"]
    label = [0, 1, 2, 3]
    cm = confusion_matrix(tl_label, pre_res, normalize='true')
    print(cm)
    print(confusion_matrix(tl_label, pre_res))
    plt.figure()
    sns.heatmap(cm, annot=True, square=True, cmap='Blues')
    plt.ylim(0, cm.shape[0])
    plt.savefig(os.path.dirname(os.getcwd()) + "/pictures/sklearn_confusion_matrix_annot_blues.png")