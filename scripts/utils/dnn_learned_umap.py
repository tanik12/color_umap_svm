import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.learning_rate = 0.0001
        #self.fc1 = nn.Linear(1728,500)
        self.fc1 = nn.Linear(1728,500)
        self.fc2 = nn.Linear(500, 2000)
        self.fc3 = nn.Linear(2000, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    #nn.Moduleにtrain()があるので別名を付与
    def train_umap(self, net, train_dataset):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=net.learning_rate)

        batch_size = 8
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        torch.manual_seed(12)
        min_loss = 10
        #for epoch in range(500):
        for epoch in range(3):
            loss = 0
            for i, train_data in enumerate(train_dataloader):
                ip, label = train_data
    
                x, y = Variable(ip), Variable(label)
                optimizer.zero_grad()
                output = net(x.float())
        
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
    
            if min_loss > loss.item():
                min_loss = loss.item()
                print(epoch, loss.item())
                torch.save(net.state_dict(), "/Users/gisen/git/color_umap_svm/model/dnn_umap.pth")

if __name__ == "__main__":
    #モデル定義
    model = Net()
