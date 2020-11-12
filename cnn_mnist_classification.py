import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import argparse  
# Parser 
parser = argparse.ArgumentParser()
parser.add_argument("-p", type=int, default=0, help="If run parallel")
args = parser.parse_args()
# Training settings
batch_size = 25000
# MNIST Dataset
train_dataset = datasets.MNIST(root='data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='data/',
                              train=False,
                              transform=transforms.ToTensor())
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(320, 10)
    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return F.log_softmax(x)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = Model().to(device=device)
# model.cuda()
##### Parallel #####
####################
####################
if bool(args.p):
    num_gpus = 2
    gpus = range(num_gpus)
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0]) 
else:
    pass
##### Parallel ####
####################
####################
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
def train(epoch):
    # start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device=device), Variable(target).to(device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(batch_idx)
        # if batch_idx % 400 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data.item()))
    # end_time = time.time()
    # print('Time Cost for Epoch {}: {:.6}'.format(
    #     epoch,
    #     end_time - start_time
    # ))
    # time.sleep(5)
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data).to(device=device), Variable(target).to(device=device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
start_time = time.time()
epoch_total = 2
for epoch in range(1, epoch_total):
    train(epoch)
    # test()
end_time = time.time()
print('Time Cost for {} Epochs: {:.6}'.format(
    epoch_total, 
    end_time - start_time
))