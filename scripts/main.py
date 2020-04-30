import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import umap
import pickle
from io import BytesIO
from PIL import Image

from skimage.color import rgb2lab
import sys
from utils.data_loader import load_image

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_model(model_name):
    try:
        with open("../model/"+ model_name +".pickle", mode='rb') as fp:
            clf = pickle.load(fp)
            return clf
    except FileNotFoundError as e:
        print("Do not exist model file! Please make model file.", e)
        sys.exit()

def train_model(model_name, feature_space="None"):
    try:
        if 'umap' in model_name:
            component_num = 3
            trained_model = umap.UMAP(n_neighbors=80, n_components=component_num, min_dist=0.4, metric='cosine',random_state=12).fit(feature_space) #たぶんこれがよい
        elif 'svm' in model_name:
            trained_model = SVC(probability=True).fit(X_train, y_train)

        return trained_model

    except FileNotFoundError as e:
        print("Do not exist model file! Please make model file.", e)
        sys.exit()

def save_model(model_name, train_obj):
    with open("../model/" + model_name + ".pickle", mode='wb') as fp:
        pickle.dump(train_obj, fp)

def color_inference(x_train, model):
    x_train = x_train.reshape(1, -1)
    pred = model.predict(x_train)
    label_name = label_dict[pred[0]]
    
    return pred, label_name

def plot(tmp, label_arrays, component_num=3):
    if component_num == 3:
        #グラフの枠を作っていく
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.set_xlabel("component1")
        ax.set_ylabel("component2")
        ax.set_zlabel("component3")

        gif_flag = False
        
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

        if gif_flag:
            for angle in range(0, 180):
                ax.view_init(30, angle*2)
                plt.savefig("../pictures/figs/{0}_{1:03d}.jpg".format("res", angle))

    elif component_num == 2:
        plt.scatter(tmp.embedding_[:, 0], tmp.embedding_[:, 1], c=label_arrays, s = 10)

    plt.show()


if __name__ == '__main__':
    ########################
    #input dataの作成
    #RGB
    img, label_arrays, image_hsv = load_image("/Users/gisen/git/color_clustering/data_evaluation")
    print("教師ラベルの総数: ", label_arrays.shape)

    #labに変換(RGB->lab)
    img = np.array(img)[:, :, :, :].astype('float32')

    img = rgb2lab(img) # lab値に変換
    img[:, :, :, 0] /= 100
    img[:, :, :, 1] /= 128
    img[:, :, :, 2] /= 128
    #さらにlab値を変換
    img[:, :, :, 0] = np.sqrt(img[:, :, :, 0])
    img[:, :, :, 1] = np.square(img[:, :, :, 1])
    img[:, :, :, 2] = np.square(img[:, :, :, 2])
    
    img_train = np.reshape(img, (len(img),-1))
    ########################

    ########################
    ######umapとsvmのobjectを作成
    # モデルの学習
    trans = train_model("model_umap", feature_space=img_train)
    X_train, X_test, y_train, y_test = train_test_split(trans.embedding_, label_arrays, test_size=0.4, random_state=12)
    svc = train_model("model_svm")

    # 学習済みモデルを保存する
    save_model("model_umap", trans)
    save_model("model_svm", svc)
    
    # 保存したモデルをロードする
    trans = load_model("model_umap")
    svc = load_model("model_svm")
    ########################

    ########################
    ######学習済みモデルへのinput data
    #X_train, X_test, y_train, y_test = train_test_split(trans.embedding_, label_arrays, test_size=0.4, random_state=12) #102~110がコメントアウトされてたらこの行はコメントアウトしないこと
    plot(trans, label_arrays) #Debug用 可視化

    ######推論
    output_proba = svc.predict_proba(X_train)
    max_index = np.argmax(output_proba, axis=1).astype('int') #predict class

    tmp = np.empty((0,4), int)
    for i in range(output_proba.shape[0]):
        idx = max_index[i]
        test = np.zeros(4).astype('int')
        test[idx] = 1
        tmp = np.append(tmp, np.array([test]), axis=0)

    max_index = max_index[:, np.newaxis]

    #ここ違う方法でやった方がいいかも
    proba_array = np.array([output_proba[i, max_index[i]] for i in range(output_proba.shape[0])])
    proba_array = np.hstack((proba_array, max_index.reshape(-1,1)))
    print(proba_array)
    ########################

    ########################
    ######評価
    print("============")
    print("dataの総数: ", X_test.shape[0])
    print("SVM:", svc.score(X_test, y_test))
    ########################

