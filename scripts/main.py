import os
import sys
import numpy as np
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split

from utils.data_loader import load_image
from utils.ml_func import load_model, train_model, save_model, color_inference
from utils.visualize import plot
from utils.evaluation import eval_confusion_matrix

##########
from utils.dnn_learned_umap import Net
from utils.data_loader_pytorch import Dataset
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
##########

def main():
    ########################
    #input dataの作成
    #RGB
    data_path = os.path.dirname(os.getcwd()) + "/data_evaluation"
    img, label_arrays, image_hsv = load_image(data_path)
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
    #img = (img[:, :, :, 0] + img[:, :, :, 1] + img[:, :, :, 2]) / 3
    
    img_train = np.reshape(img, (len(img),-1))
    ########################

    ########################
    #train, test dataに分割
    X_train, X_test, y_train, y_test = train_test_split(img_train, label_arrays, test_size=0.2, random_state=19)
    ########################

    ########################
    ######umapとsvmのobjectを作成
    # モデルの学習
    trans = train_model("model_umap", train_data=X_train)
    X_train222 = trans.embedding_
    y_train = y_train

    svc = train_model("model_svm", train_data=X_train222, tl_data=y_train)

    # 学習済みモデルを保存する
    save_model("model_umap", trans)
    save_model("model_svm", svc)
    
    # 保存したモデルをロードする
    trans = load_model("model_umap")
    svc = load_model("model_svm")

    #test dataの作成
    X_test222 = trans.transform(X_test)
    ########################

    ########################
    ######推論
    output_proba = svc.predict_proba(X_test222)
    max_index = np.argmax(output_proba, axis=1).astype('int') #predict class

    tmp = np.empty((0,4), int)
    for i in range(output_proba.shape[0]):
        idx = max_index[i]
        test = np.zeros(4).astype('int')
        test[idx] = 1
        tmp = np.append(tmp, np.array([test]), axis=0)

    max_index_pre = max_index
    max_index = max_index[:, np.newaxis]
    #ここ違う方法でやった方がいいかも
    proba_array = np.array([output_proba[i, max_index[i]] for i in range(output_proba.shape[0])])
    proba_array = np.hstack((proba_array, max_index.reshape(-1,1)))
    #print(proba_array) #debug
    ########################

    ####################
    ##評価
    print("============")
    print("test_dataの総数: ", X_test222.shape[0])
    print("SVM:", svc.score(X_test222, y_test))
    eval_confusion_matrix(y_test, max_index_pre)
    ####################

    ########################
    #umap空間を学習させるための準備
    #train data用のデータ
    input = torch.from_numpy(X_train)
    target = torch.from_numpy(X_train222)
    
    #test data用のデータ
    input_test = torch.from_numpy(X_test)
    target_test = torch.from_numpy(y_test)
    train_dataset = Dataset(input, target, transform=None)

    ### umap空間をDNNにより学習
    net = Net()
    net.train_umap(net, train_dataset)

    ### model load
    net.load_state_dict(torch.load("/Users/gisen/git/color_umap_svm/model/dnn_umap.pth"))
    net.eval()

    ### DNNにより推論
    outputs = net(input_test.float())
    outputs = outputs.to('cpu').detach().numpy().copy()
    plot(outputs, np.array(target_test))
    plot(X_test222, np.array(target_test))
    ########################

if __name__ == '__main__':
    main()
