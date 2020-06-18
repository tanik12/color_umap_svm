import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_image(image_file):
    file_name_lists = os.listdir(image_file)
    label_arrays = []

    # cv2 load images as BGR
    image_bgr = []
    for i in file_name_lists:
        img = cv2.imread(image_file+'/'+i)
        height, width, channels = img.shape[:3]
        #if height >= 75 or width >= 75:
        if height >= 24 or width >= 24:
            image_bgr.append(img)
            label_arrays.append(i)
        else:
            continue
        
    label_arrays = color_label_list(label_arrays)

    image_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in image_bgr]
    image_rgb = [cv2.resize(i, (24, 24)).astype(int)  for i in image_rgb]

    image_hsv = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in image_bgr]
    image_hsv = [cv2.resize(i, (24, 24)).astype(int)  for i in image_hsv]

    return image_rgb, label_arrays, image_hsv

def color_label_list(file_name_lists):
    labels = np.array([])
    for img_label in file_name_lists:
        if 'red' in img_label:
            labels = np.append(labels, 0)
        if 'blue' in img_label:
            labels = np.append(labels, 1)
        if 'yellow' in img_label:
           labels =  np.append(labels, 2)
        if 'unknown' in img_label:
            labels = np.append(labels, 3)
    return labels

if __name__ == "__main__":
    img = load_image(os.path.dirname(os.getcwd()) + "/data")
    #img = load_image(os.path.dirname(os.getcwd()) + "/data_evaluation")
    print(type(img))
    img=np.reshape(img, (len(img),150,150,3))
    print(img.shape)
    hstack=np.hstack(img)
    print(hstack.shape)
    plt.imshow(hstack)
    plt.show()
