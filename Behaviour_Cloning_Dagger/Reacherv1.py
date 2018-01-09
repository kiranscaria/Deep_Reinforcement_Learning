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
        self.fc1 = nn.Linear(11, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

ITERATIONS = 1000
BATCH_SIZE = 5000
X = [] 
Y = []

def train(dataset):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # get the inputs
    in_data, out_data = dataset['observations'][:], dataset['actions'][:,0,:]

    for epoch in range(ITERATIONS): # loop over the dataset multiple times

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
            
        print('[%d] loss:%.3f' % (epoch+1, running_loss))
        X.append(epoch+1), Y.append(running_loss)   

    # save the model
    torch.save(net.state_dict(), 'reacher_model.pth.tar')

    print('Finished Training')

    return X, Y
    
def test(x):
    in_data = torch.from_numpy(x).float()
    out_data = net(Variable(in_data))
    return out_data.data.numpy()

def main():
    train(expert_data)

if __name__ == '__main__':
    main()
