import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 784  #28*28
num_classes = 10

num_epochs = 5
batch_size = 60
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


def leaky_relu(mat):
    zeros = mat - mat
    positives = torch.maximum(mat, zeros)
    negatives = torch.minimum(mat * 0.6, zeros)
    return positives + negatives


class TwoLayers(nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayers, self).__init__()
        self.linear1 = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(350, 100)
        self.linear3 = nn.Linear(100, output_size)

    def forward(self, x):
        out1 = self.linear1(x)
        return self.linear3(leaky_relu(out1))


model = TwoLayers(input_size, num_classes)


def my_loss(output, target):
    mat = torch.tensor([
        [0.85, 0, 0, 0, 0, 0, 0, 0, 0.02, 0],  # 0
        [0, 0.9, 0, 0, 0, 0, 0, 0.1, 0, 0],  # 1
        [0, 0, 0.95, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0., 0, 0, 0.9, 0, 0, 0, 0, 0.02, 0],  # 3
        [0., 0, 0, 0, 0.9, 0, 0, 0, 0.0, 0],  # 4
        [0., 0, 0, 0, 0, 0.95, 0, 0, 0.02, 0],  # 5
        [0.05, 0, 0.05, 0.02, 0, 0, 0.9, 0, 0.02, 0],  # 6
        [0., 0.1, 0, 0, 0, 0, 0, 0.9, 0., 0],  # 7
        [0.05, 0, 0, 0.1, 0, 0, 0.1, 0, 0.9, 0.1],  # 8
        [0.05, 0, 0, 0, 0.08, 0.05, 0, 0, 0.02, 0.9],  # 9
    ])

    mat = torch.diag(torch.ones(10))

    res = 0
    for (pred, cls) in zip(output, target):
        softmax_ = nn.Softmax(dim=0)(pred)
        log_softmax = torch.log(softmax_)
        mat_cls_ = mat[:, cls]
        res_ = (log_softmax * mat_cls_).sum()
        # res_ = log_softmax[cls]
        # print("softmax: ", softmax_, softmax_.sum(), "\nlog_softmax: ", log_softmax, "\nmat_cls: ", mat_cls_, "\nres: ", res_)

        res = res - res_
    # print(res)
    return res / batch_size


# Loss and Optimizer
# Softmax is internally computed.
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.85)

cnt = 0

for epoch in range(num_epochs):
    # if cnt > 0:
    #     break
    for i, (images, labels) in enumerate(train_loader):
        # if cnt > 1:
        #     break
        images = images.view(-1, 28 * 28)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = ce_loss(outputs, labels)
        # exit(-1)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4}'.format(epoch + 1, num_epochs,
                                                                  i + 1, len(mnist_train) // batch_size,
                                                                  loss.item()))

correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    outputs = model(images)
    predicted = torch.argmax(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: ', float(correct) / total)
