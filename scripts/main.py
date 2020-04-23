import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from skimage.color import rgb2lab
import sys
from utils.data_loader import load_image

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot(tmp, label_arrays, component_num):
    if component_num == 3:
        #グラフの枠を作っていく
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.set_xlabel("component1")
        ax.set_ylabel("component2")
        ax.set_zlabel("component3")
        
        for idx, label in enumerate(label_arrays):
            label = label.astype(int)
            if label == 0:
                ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="red", s = 10)
            elif label == 1:
                ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="blue", s = 10)
            elif label == 2:
                ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="yellow", s = 10)
            else:
                ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="black", s = 10)
    elif component_num == 2:
        plt.scatter(tmp.embedding_[:, 0], tmp.embedding_[:, 1], c=label_arrays, s = 10)

    plt.show()


if __name__ == '__main__':
    img, label_arrays, image_hsv = load_image("/Users/gisen/git/color_clustering/data_evaluation")
    ####
    #RGB
    print("教師ラベルの総数: ", label_arrays.shape)
    ###img = np.reshape(img, (len(img),50,50,3))
    ####

    ##########
    ###hsv
    ##print("aaaa:", np.array(img)[:, :, :, :3].shape)
    ##img_hsv = np.array(img)[:, :, :, :3].astype('float32')
    ##img_hsv[:, :, :, 0] /= 180
    ###img_hsv[:, :, :, 2] /= 255
    ###img_hsv[:, :, :, 1] = img_hsv[:, :, :, 2]
    ##img_hsv = np.array(img_hsv)[:, :, :, 0]
##
    ##img_train_hsv = np.reshape(img_hsv, (len(img_hsv),-1))
    ############

    #######
    #lab
    img = np.array(img)[:, :, :, :].astype('float32')

    ###image_hsv = np.array(image_hsv)[:, :, :, :1].astype('float32')
    ###print(np.median(img[:, :, :, 0]))
    img = rgb2lab(img) # LAB値に変換
    img[:, :, :, 0] /= 100
    img[:, :, :, 1] /= 128
    img[:, :, :, 2] /= 128
    img[:, :, :, 0] = np.sqrt(img[:, :, :, 0])
    img[:, :, :, 1] = np.square(img[:, :, :, 1])
    img[:, :, :, 2] = np.square(img[:, :, :, 2])

    #img = img[:, :, :, 1:3]    
    
    img_train = np.reshape(img, (len(img),-1))
    #######

    #bar_lab = rgb2lab(img) # LAB値に変換
    print(img_train.shape)
    #img_train = np.reshape(bar_lab, (len(bar_lab),-1))
    #print(img_train.shape)
    
    import umap
    # sklearnと同じようなインターフェイス
    #n_neighborsを変えると結果が結構変わる
    #処理時間を測るために%timeをしている
    component_num = 3
    ###trans = umap.UMAP(n_neighbors=12, n_components=component_num, random_state=12).fit(img_train)
    ##trans = umap.UMAP(n_neighbors=40, n_components=component_num, random_state=12).fit(img_train) #たぶんこれがよい 12
    trans = umap.UMAP(n_neighbors=80, n_components=component_num, min_dist=0.4, metric='cosine',random_state=12).fit(img_train) #たぶんこれがよい
    #trans = umap.UMAP(n_neighbors=30, n_components=component_num, metric='correlation', random_state=12).fit(img_train)
    plot(trans, label_arrays, component_num)
    ###plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=label_arrays, cmap='Spectral')
    ###plt.title('Embedding of the training set by UMAP', fontsize=24)
    ###plt.show()

    X_train, X_test, y_train, y_test = train_test_split(trans.embedding_, label_arrays, test_size=0.4, random_state=12)

    svc = SVC().fit(X_train, y_train)
    print("SVM:", svc.score(X_test, y_test))

    ###svc = SVC().fit(trans.embedding_, label_arrays)
    ###print("SVM:", svc.score(trans.embedding_, label_arrays))

