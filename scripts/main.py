import sys
import numpy as np
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split

from utils.data_loader import load_image
from utils.ml_func import load_model, train_model, save_model, color_inference
from utils.visualize import plot

def main():
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
    trans = train_model("model_umap", train_data=img_train)
    X_train, X_test, y_train, y_test = train_test_split(trans.embedding_, label_arrays, test_size=0.4, random_state=12)
    svc = train_model("model_svm", train_data=X_train, tl_data=y_train)

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

if __name__ == '__main__':
    main()
