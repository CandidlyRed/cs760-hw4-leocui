import numpy as np
from torchvision import datasets, transforms
from scipy.special import softmax, expit
import matplotlib.pyplot as plt

# for q3 and q4
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

mTrainset = datasets.MNIST(
    root="./hw4Data2", 
    download=True, 
    train=True, 
    transform=transforms.ToTensor()
    )
mTestset = datasets.MNIST(
    root="./hw4Data2", 
    download=True, 
    train=False, 
    transform=transforms.ToTensor()
    )

def reshapeData(data):
    X = np.zeros(len(data) * 28 * 28).reshape(len(data), 784)
    for i in range(len(data)):
        X[i] = np.array(data[i][0]).reshape(1, 784)[0]
    return(X.T)

def getLabel(data):
    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i][1]
    return(y)

xTrain,yTrain = reshapeData(mTrainset),getLabel(mTrainset)
xTest,yTest = reshapeData(mTestset),getLabel(mTestset)

def predict(data, W1, W2):
    return(softmax(np.matmul(W2, expit(np.matmul(W1, data)))))

def calcLoss(labels, predictions):
    loss = []
    for i in range(len(labels)):
        index = int(labels[i] - 1)
        yHat = predictions[index, i]
        loss.append(np.log(yHat))
    return(-sum(loss))

def oneHotVectors(labels):
    mat = np.zeros((200 * len(labels))).reshape(200, len(labels))
    for i in range(len(labels)):
        index = int(labels[i] - 1)
        mat[index, i] = 1
    return(mat)

# q2

# W1 = np.random.uniform(low=0, high=1, size=(784 * 300)).reshape(300, 784)
# W2 = np.random.uniform(low=0, high=1, size=(300 * 200)).reshape(200, 300)

# h = .0001
# epochs = 10
# it = 1
# batchSize = 60
# y = oneHotVectors(yTrain)

# trainLoss = [calcLoss(yTrain, predict(xTrain, W1, W2)) / 60000]
# testLoss = [calcLoss(yTest, predict(xTest, W1, W2)) / 10000]
# error = []

# while it <= epochs:
#     batchId = np.random.choice(range(len(yTrain)), size=60000, replace=False)
#     batches = len(yTrain) / batchSize
#     for i in range(int(batches)):
#         start = i * batchSize
#         stop = start + batchSize
#         ids = batchId[start:stop]
#         xBatch = xTrain[:, ids]
#         yBatch = y[:, ids]
#         a = expit(np.matmul(W1, xBatch))
#         d = np.matmul(W2.T, (predict(xBatch, W1, W2) - yBatch)) * a * (1 - a)
#         W2 = W2 - h * np.matmul((predict(xBatch, W1, W2) - yBatch), a.T)
#         W1 = W1 - h * np.matmul(d, xBatch.T)
#     trainLoss.append(calcLoss(yTrain, predict(xTrain, W1, W2)) / 60000)
#     testLoss.append(calcLoss(yTest, predict(xTest, W1, W2)) / 10000)
#     pred = predict(xTest, W1, W2)
#     e = 0
#     for i in range(len(yTest)):
#         p = np.argmax(pred[:, i])
#         if p != int(yTest[i]): e += 1
#     error.append(e / len(yTest))
#     it += 1

# for i in range(len(error)):
#     print(i+1, round(error[i]*100, 4))
# plt.plot(range(epochs + 1), trainLoss, label='train loss')
# plt.plot(range(epochs + 1), testLoss, label='test loss')
# plt.legend(loc="upper right")
# plt.title("Learning Curve")
# plt.xlabel("epoch")
# plt.ylabel("Average Loss")
# plt.show()

trainLoss = []
testLoss = []

testErrors = []

mTrainset = datasets.MNIST(
    root="./hw4Data2",  
    download=True, 
    train=True, 
    transform=transforms.ToTensor(), 
    target_transform=lambda y: torch.tensor(y, dtype=torch.long)
    )
mTestset = datasets.MNIST(
    root="./hw4Data2", 
    download=True, 
    train=False, 
    transform=transforms.ToTensor(),
    target_transform=lambda y: torch.tensor(y, dtype=torch.long)
    )

class NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.w1 = nn.Linear(28*28, 300)
        self.w2 = nn.Linear(300, 200)
        nn.init.uniform_(self.w1.weight, a=-1, b=1)
        nn.init.uniform_(self.w1.weight, a=-1, b=1)
        self.stack = nn.Sequential(
            self.w1,
            nn.Sigmoid(),
            self.w2,
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

model = NN()

def training(dataloader, model, lossFunc, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = lossFunc(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
    
    trainLoss.append(loss.item())
    
def testing(dataloader, model, lossFunc):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += lossFunc(pred, y).item()
            correct += (pred.argmax(dim=1) == y).sum().item()

    test_loss /= num_batches
    testLoss.append(test_loss)
    correct /= size
    testErrors.append(correct)

trainDataLoader = DataLoader(mTrainset, batch_size=32, shuffle=True)
testDataLoader = DataLoader(mTestset, batch_size=32, shuffle=True)
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



for i in range(20):
    training(trainDataLoader, model, lossFunc, optimizer)
    testing(testDataLoader, model, lossFunc)

for i in range(20):
    print(i+1, "&", round((1 - testErrors[i])*100, 4), "\%", "\\\\")

plt.plot(range(len(trainLoss)), trainLoss, label='train loss')
plt.plot(range(len(testLoss)), testLoss, label='test loss')
plt.legend(loc="upper right")
plt.title("Learning Curve")
plt.xlabel("epoch")
plt.ylabel("Average Loss")
plt.show()
