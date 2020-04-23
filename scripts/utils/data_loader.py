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
        ###img = cv2.imread(image_file+'/'+i, cv2.IMREAD_GRAYSCALE)
        ###height, width = img.shape
        if height >= 75 or width >= 75:
            image_bgr.append(img)
            label_arrays.append(i)
        else:
            continue
        
    label_arrays = color_label_list(label_arrays)

    ###image_rgb = [cv2.resize(i, (50, 50)) for i in image_bgr]
    image_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in image_bgr]
    #####image_rgb = [cv2.resize(i, (50, 50)).astype('float32') / 255.  for i in image_rgb] COLOR_Lab2RGB
    #####image_rgb = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in image_bgr]
    #####image_rgb = [cv2.cvtColor(i, cv2.COLOR_Lab2RGB) for i in image_bgr]
    image_rgb = [cv2.resize(i, (24, 24)).astype(int)  for i in image_rgb]

    image_hsv = [cv2.cvtColor(i, cv2.COLOR_BGR2HSV) for i in image_bgr]
    image_hsv = [cv2.resize(i, (24, 24)).astype(int)  for i in image_hsv]

    return image_rgb, label_arrays, image_hsv

def color_list():
    red_lab    = np.array([56, 77, 32])    #lba空間における赤信号の値
    #red_lab_2  = np.array([84, 15, 22])    #lba空間における赤信号の値
    #red_lab_3  = np.array([39, 62, 30])    #lba空間における赤信号の値
    
    aqua_lab   = np.array([91, -48, -10])  #lba空間における青信号の値
    #aqua_lab_2 = np.array([88, -26, -4])  #lba空間における青信号の値
    green_lab  = np.array([73, -61, 30])   #lba空間における青信号の値
    #green_lab_2  = np.array([41, -28, 2])   #lba空間における青信号の値

    yellow_lab = np.array([86, 0, 86])    #lba空間における黄信号の値
    ###yellow_lab = np.array([86, -7, 86])    #lba空間における黄信号の値
    #yellow_lab_2 = np.array([89, -3, 59])    #lba空間における黄信号の値
    #yellow_lab_3 = np.array([76, 19, 76])    #lba空間における黄信号の値

    #unknown_lab = np.array([76, 0, -4])    #lba空間におけるunknown信号の値
    #unknown_lab_2 = np.array([81, -6, -8])    #lba空間におけるunknown信号の値
    #unknown_lab_3 = np.array([56, 1, -9])    #lba空間におけるunknown信号の値

    colors_lab = np.vstack((red_lab, aqua_lab, green_lab, yellow_lab))
    return colors_lab

def color_list_():
    red_lab    = np.array([56, 77, 32])    #lba空間における赤信号の値
    #red_lab_2  = np.array([84, 15, 22])    #lba空間における赤信号の値
    red_lab_3  = np.array([39, 62, 30])    #lba空間における赤信号の値
    
    aqua_lab   = np.array([91, -48, -10])  #lba空間における青信号の値
    #aqua_lab_2 = np.array([88, -26, -4])  #lba空間における青信号の値
    green_lab  = np.array([73, -61, 30])   #lba空間における青信号の値
    #green_lab_2  = np.array([41, -28, 2])   #lba空間における青信号の値

    yellow_lab = np.array([86, -7, 86])    #lba空間における黄信号の値
    #yellow_lab_2 = np.array([89, -3, 59])    #lba空間における黄信号の値
    #yellow_lab_3 = np.array([76, 19, 76])    #lba空間における黄信号の値

    #unknown_lab = np.array([76, 0, -4])    #lba空間におけるunknown信号の値
    unknown_lab_2 = np.array([81, -6, -8])    #lba空間におけるunknown信号の値
    #unknown_lab_3 = np.array([56, 1, -9])    #lba空間におけるunknown信号の値

    colors_lab = np.vstack((red_lab, red_lab_3, aqua_lab, green_lab, yellow_lab, unknown_lab_2))
                            #yellow_lab, yellow_lab_2, yellow_lab_3, unknown_lab, unknown_lab_2, unknown_lab_3))
    return colors_lab

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
    img = load_image("/Users/gisen/git/color_clustering/data")
    #img = load_image("/Users/gisen/git/color_clustering/data/data_evaluation")
    print(type(img))
    img=np.reshape(img, (len(img),150,150,3))
    print(img.shape)
    hstack=np.hstack(img)
    print(hstack.shape)
    plt.imshow(hstack)
    plt.show()
