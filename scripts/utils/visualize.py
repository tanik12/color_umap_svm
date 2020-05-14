import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(tmp, label_arrays, component_num=2):
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
                ax.scatter(tmp[idx, 0], tmp[idx, 1], tmp[idx, 2], c="red", s = 10)
                #ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="red", s = 10)
            elif label == 1:
                ax.scatter(tmp[idx, 0], tmp[idx, 1], tmp[idx, 2], c="blue", s = 10)
                #ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="blue", s = 10)
            elif label == 2:
                ax.scatter(tmp[idx, 0], tmp[idx, 1], tmp[idx, 2], c="yellow", s = 10)
                #ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="yellow", s = 10)
            else:
                ax.scatter(tmp[idx, 0], tmp[idx, 1], tmp[idx, 2], c="black", s = 10)
                #ax.scatter(tmp.embedding_[idx, 0], tmp.embedding_[idx, 1], tmp.embedding_[idx, 2], c="black", s = 10)

        if gif_flag:
            for angle in range(0, 180):
                ax.view_init(30, angle*2)
                plt.savefig("../pictures/figs/{0}_{1:03d}.jpg".format("res", angle))

    elif component_num == 2:
        plt.scatter(tmp[:, 0], tmp[:, 1], c=label_arrays, s = 10)
        #plt.scatter(tmp.embedding_[:, 0], tmp.embedding_[:, 1], c=label_arrays, s = 10)

    plt.show()
    plt.close()

if __name__ == "__main__":
    pass
