## build CNN
from torch import nn

class Net(nn.Module):                 
    def __init__(self):
        super(Net, self).__init__()   
        self.fc1 = nn.Linear(13,256) 
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(256,256)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(256,2)
        
    def forward(self, input):
            output = input.view(-1, 13)
            output = self.fc1(output)
            output = self.relu1(output)
            output = self.fc2(output)
            output = self.relu2(output)
            output = self.fc3(output)
            
            return output