import numpy as np
import torch
import torch.nn as nn
from utils.dnn_inference_umap import ColorNet

def main():
    #モデル定義
    net = ColorNet()
    net.load_state_dict(torch.load("/Users/gisen/git/color_umap_svm/model/dnn_umap.pth"))
    net.eval()

    #data load
    input_test = np.load('test_data.npy')
    input_test = torch.from_numpy(input_test)
    
    outputs = net(input_test.float())    
    outputs = outputs.to('cpu').detach().numpy().copy()
    print(outputs)

    pre = net.inference_color(outputs)
    print(pre)
    
if __name__ == '__main__':
    main()
    #test data
    #tmp = np.array([[10.845396,   7.6633716,  3.7420197],
    #                [ 8.210488,  10.083481,  12.812289 ],
    #                [ 8.273848,   7.998415,   0.9959157],
    #                [10.340503,   8.614014,   5.1347723],
    #                [ 7.1470094,  8.76033,    1.0080775]])
