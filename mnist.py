import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 784  #28*28
num_classes = 10

num_epochs = 5
batch_size = 100
learning_rate = 0.005

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.3081,],std=[0.1306,])])
mnist_train = datasets.MNIST("./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST("./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=False)


def tanh(t):
    return torch.tanh(t)


def tanhPrime(t):
    # derivative of tanh
    # t: tanh output
    res = 1 - t * t
    return res


# our version to softmax
def softmax(vector, dim=None, keepdim=None):
    powered = torch.exp(vector)
    if dim is not None and keepdim is not None:
        summed = torch.sum(powered, dim=dim, keepdim=keepdim)
    else:
        summed = torch.sum(powered)

    return powered / summed


# our version to CE
def cross_entropy_loss(output, target):
    mat = torch.diag(torch.ones(10))
    res = 0

    for (pred, cls) in zip(output, target):
        softmax_ = softmax(pred)
        log_softmax = torch.log(softmax_)
        mat_cls_ = mat[:, cls]
        res_ = (log_softmax * mat_cls_).sum()

        res = res - res_

    return res / batch_size


class Neural_Network:
    def __init__(self, input_size=784, output_size=10, hidden_size=100):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)
        self.cnt = 0

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = tanh(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        res = softmax(self.z2, 1, True)

        return res

    def backward(self, X, y, y_hat, lr=.5):
        dl_dz2 = (1 / batch_size) * (y_hat - y)
        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * tanhPrime(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

        # print("back:", time.perf_counter() - t_)


    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)


def print_loss_to_screen():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        out = NN.forward(images)
        predicted = torch.argmax(out, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print(f"acc out of : {total}", float(correct) / total)


NN = Neural_Network()
for epoch in range(num_epochs):

    for i, (x, y) in enumerate(train_loader):
        x = x.view(-1, 784)
        # turn label to one-hot
        y_ = torch.zeros(batch_size, num_classes).scatter_(1, y.unsqueeze(1), 1)

        NN.train(x, y_)
        # if i % 100 == 0:
        #     y_hat = NN.forward(x)
        #     print(f"epoch: #{epoch}, step: #{i}, Loss: {cross_entropy_loss(y_hat, y)}")

    print(f"epoch: #{epoch}")
    print_loss_to_screen()

