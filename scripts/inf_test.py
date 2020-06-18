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
    img = np.load('test_img.npy')

    #dimension compression. express umap as dnn.
    input_img = net.input_preprocessing(img)
    input_test = torch.from_numpy(input_img)
    outputs = net(input_test.float())    
    outputs = outputs.to('cpu').detach().numpy().copy()
    
    #color inference
    pre = net.inference_color(outputs)
    print(pre)
    
if __name__ == '__main__':
    main()
