import numpy as np
import torch #基本モジュール
import torch.nn as nn #ネットワーク構築用
from utils.ml_func import load_model
import torch.nn.functional as F #ネットワーク用の様々な関数
from skimage.color import rgb2lab

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.learning_rate = 0.001
        self.fc1 = nn.Linear(1728,100)
        self.fc2 = nn.Linear(100,25)
        self.drop_2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(25, 100)
        self.drop_3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(100, 3)

        self.color_model = self.get_color_model()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_2(x)        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop_3(x)        
        x = self.fc4(x)
        return x

    def get_color_model(self):
        color_model = load_model("model_svm")
        return color_model

    def input_preprocessing(self, img): #imgのshape -->> (n, 24, 24, 3)
        #labに変換(RGB->lab)
        img = np.array(img)[:, :, :, :].astype('float32')
        img = rgb2lab(img) # lab値に変換
        #正規化処理
        img[:, :, :, 0] /= 100
        img[:, :, :, 1] /= 128
        img[:, :, :, 2] /= 128
        #さらにlab値を変換
        img[:, :, :, 0] = np.sqrt(img[:, :, :, 0])
        img[:, :, :, 1] = np.square(img[:, :, :, 1])
        img[:, :, :, 2] = np.square(img[:, :, :, 2])
        #img = (img[:, :, :, 0] + img[:, :, :, 1] + img[:, :, :, 2]) / 3
        
        input_data = np.reshape(img, (len(img),-1))

        return input_data  #imgのshape -->> (n, 1728)

    def inference_color(self, X_test222):
        output_proba = self.color_model.predict_proba(X_test222)
        max_index = np.argmax(output_proba, axis=1).astype('int') #predict class
    
        max_index = max_index[:, np.newaxis]
        #ここ違う方法でやった方がいいかも
        proba_array = np.array([output_proba[i, max_index[i]] for i in range(output_proba.shape[0])])
        proba_array = np.hstack((proba_array, max_index.reshape(-1,1)))
        #print(proba_array) #debug
        return proba_array

if __name__ == "__main__":
    #モデル定義
    net = ColorNet()

    #data load
    input_test = np.load('test_data.npy')
    outputs = net(input_test.float())
    outputs = outputs.to('cpu').detach().numpy().copy()

    #test data
    tmp = np.array([[10.845396,   7.6633716,  3.7420197],
                    [ 8.210488,  10.083481,  12.812289 ],
                    [ 8.273848,   7.998415,   0.9959157],
                    [10.340503,   8.614014,   5.1347723],
                    [ 7.1470094,  8.76033,    1.0080775]])
    
    model.inference_color(outputs)