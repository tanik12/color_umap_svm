import sys
from sklearn.svm import SVC
import umap
import pickle

def load_model(model_name):
    try:
        with open("../model/"+ model_name +".pickle", mode='rb') as fp:
            clf = pickle.load(fp)
            return clf
    except FileNotFoundError as e:
        print("Do not exist model file! Please make model file.", e)
        sys.exit()

def train_model(model_name, train_data="None", tl_data="None"):
    try:
        if 'umap' in model_name:
            component_num = 3
            trained_model = umap.UMAP(n_neighbors=80, n_components=component_num, min_dist=0.4, metric='cosine',random_state=12).fit(train_data) #たぶんこれがよい
            ####trained_model = umap.UMAP(n_neighbors=80, n_components=component_num, min_dist=0.4, metric='cosine',random_state=12).fit(train_data) #たぶんこれがよい
        elif 'svm' in model_name:
            trained_model = SVC(probability=True).fit(train_data, tl_data)

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

if __name__ == "__main__":
    pass
