import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


def view_classify(img, ps):
    ps = ps.cpu().data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# image converter to tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# get dataset
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
# create loaders with datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#  setup interator for all images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#  build a image to show example numbers
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
plt.show()

# net variables
input_size = 784  # input values number of pixels in each image
hidden_sizes = [128, 64]  # Two hidden layers first with 126 and the second with 64
output_size = 10  # output value from 0 - 9

# build model
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),  # connect input to hidden layer1
                      nn.ReLU(),  # positive values pass through negative values set to zero
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # connect hidden layer1 to hidden layer2
                      nn.ReLU(),   # positive values pass through negative values set to zero
                      nn.Linear(hidden_sizes[1], output_size),  # connect hidden layer2 to output
                      nn.LogSoftmax(dim=1))  # use softlog for passing values between nodes

criterion = nn.NLLLoss()  # lossless backprop
images, labels = next(iter(trainloader))  # images and their lables
images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)  #perform gradient descent and update with backprop
time0 = time()  # time before we started
epochs = 15  # number of runs
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)  # Flatten MNIST images into a 784 long vector

        optimizer.zero_grad()  # Training pass

        output = model(images)  # did the model think it saw
        loss = criterion(output, labels)  # how wrong was it

        loss.backward()  # This is where the model learns by backpropagating

        optimizer.step()  # And optimizes its weights here

        running_loss += loss.item()  # get the total wrongness
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

#  How long did it take to train
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

#  Get next image
images, labels = next(iter(valloader))

#  pick first image
img = images[0].view(1, 784)
#  run image through model
with torch.no_grad():
    logps = model(img)

# get result
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))

# display listed value
view_classify(img.view(1, 28, 28), ps)

