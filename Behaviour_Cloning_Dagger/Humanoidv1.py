import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import pickle
import matplotlib.pyplot as plt

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # input size : 11 ; output size : 2; 2 layers
        self.fc1 = nn.Linear(376, 2000)
        self.fc2 = nn.Linear(2000, 5000)
        self.fc3 = nn.Linear(5000, 1000)
        self.fc4 = nn.Linear(1000, 17)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()

BATCH_SIZE = 250
X = [] 
Y = []

def train(dataset):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00003, momentum=0.9)

    # get the inputs
    in_data, out_data = dataset['observations'][:], dataset['actions'][:,0,:]

    for epoch in range(10): # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(int(in_data.shape[0]/BATCH_SIZE)):
            inputs, outputs = in_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE][:], out_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE][:]

            # wrap them in Variable 
            inputs, outputs = Variable(torch.from_numpy(inputs).float()), Variable(torch.from_numpy(outputs).float())

	    # zero the parameter gradients
            optimizer.zero_grad()

	    # forward + backward + optimize
            predictions = net(inputs)

            loss = criterion(predictions, outputs)  
            loss.backward()
            optimizer.step()

	    # print statistics
            running_loss += loss.data[0]
            
        print('[%d] loss:%.3f' % (epoch+1, running_loss/BATCH_SIZE))
        X.append(epoch+1), Y.append(running_loss/BATCH_SIZE)   

    print('Finished Training')

def test(x):
    in_data = torch.from_numpy(x).float()
    out_data = net(Variable(in_data))
    return out_data.data.numpy()
    
def main():
    expert_data = pickle.load(open("networks/Humanoid-v1.p", "rb"))       
    print('observations : ', expert_data['observations'][:].shape)
    print('actions : ', expert_data['actions'][:,0,:].shape)
    train(expert_data)

if __name__ == '__main__':
    main()
