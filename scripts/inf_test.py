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
    outputs_3dim = net.get_color_feature(net, img)
    #color inference
    pre = net.inference_color(outputs_3dim)
    print(pre)
    
if __name__ == '__main__':
    main()
