import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters -- feel free to play around with these
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Number of training batches to go over before we test the model
TEST_FREQUENCY = 100

# The following code downloads the MNIST Dataset and creates train/test data loaders.
# This dataset contains a large collection of 28x28 grayscale images corresponding to
# handwritten digits (0-9). We apply a transformation to standardize the dataset, which
# helps to remove biases on certain features and keep input data in a consistent format.
# The mean (0.1307) and std. deviation (0.3081) are well defined for this data set, which
# is why they are directly used in the normalization below.

# PyTorch has a number of datasets that can be imported easily using the
# torchvision.datasets module. See https://pytorch.org/vision/0.15/datasets.html.
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# TODO: Visualize 6 of the data samples from the testing dataset in a 2x3 grid
# along with the corresponding digit value as a label. You can follow
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html to see how
# to get a batch of images and the corresponding labels from the data loaders.



# Define the CNN architecutre. One such architecture (LeNet-5) is described
# in the comments below. However, feel free to research and implement your own model.
# The LeNet-5 architecture was proposed in http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf.
# We will be adapting this to take a 28x28x1 image as input. It consists of two sets of convolutional
# and max-pooling layers followed by two fully connected layers and an output layer. It also uses the
# ReLU activation function at multiple instances -- see the comments in the forward() function.
# The output of this network is 10 values, corresponding to the probability of the input
# being the digit 0, 1, 2, ..., 9 respectively. So your final prediction for an input, will
# be the index corresponding to the highest probability.
class Net(nn.Module):
    # Define the layers for LeNet-5, TODO: Replace ___ with corresponding values
    def __init__(self):
        super().__init__()
        # Convolutional Layer 1: Input channels=1, Output channels=6, Kernel size=5x5
        self.conv1 = nn.Conv2d(___, ___, kernel_size=___)
        # Max-Pooling Layer: Kernel size=2x2, Stride=2
        self.pool = nn.MaxPool2d(kernel_size=___, stride=___)
        # Convolutional Layer 2: Output channels=16, Kernel size=5x5
        self.conv2 = nn.Conv2d(___, ___, kernel_size=___)
        # Fully Connected Layer 1: Input features=?, Output features=120
        # TODO: Compute # of input features, hint: this corresponds to the total
        # number of pixels in the 16 channels after the second max pooling layer
        self.fc1 = nn.Linear(___, ___)
        # Fully Connected Layer 2: Input features=120, Output features=84
        self.fc2 = nn.Linear(___, ___)
        # Fully Connected Layer 3 (Output Layer): Input features=84, Output features=10
        self.fc3 = nn.Linear(___, ___)

    # Implement the forward pass for LeNet-5, TODO: Replace ___ with corresponding implementation
    # using the modules you initialized above. The first one has been done for you.
    def forward(self, x):
        # Apply Conv1 followed by ReLU and Max-Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply Conv2 followed by ReLU and Max-Pooling
        x = ___
        # Flatten the feature maps
        x = torch.flatten(x, 1) # flattens all dimensions except batch
        # Apply FC1 followed by ReLU
        x = ___
        # Apply FC2 followed by ReLU
        x = ___
        # Apply FC3 (output layer)
        x = ___
        # Return the result of network
        return x

# Initialize the network, loss function (cross entropy), and optimizer (SGD)
net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# TODO: Visualize 6 of the data samples from the testing dataset in a 2x3 grid
# along with the corresponding predicted digit value as a label. You will need to
# pass these images through the network, and then interpret the result to get the
# prediction. Helpful function: torch.max()

# Define a function for testing the network
def test(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # TODO: Pass the inputs through your network, compute predictions,
            # and compare them to provided labels. Update the number of correct
            # predictions and the number of total predictions.
            raise NotImplementedError

    accuracy = 100 * correct / total
    return accuracy

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    for batch_no, (inputs, labels) in enumerate(train_loader):
        # TODO: Zero out the gradients of the optimizer to prepare it for the next
        # set of data, pass the inputs through the network, compute the loss between
        # the outputs and provided labels, perform backward propagation using the loss term,
        # and perform the optimization step. Hint: this can all be done in 5 lines of code,
        # do not try to implement anything from scratch.
        raise NotImplementedError

        if (batch_no % TEST_FREQUENCY == 0):
            accuracy = test(net, test_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_no}/{len(train_loader)}, Accuracy: {accuracy:.2f}%")
accuracy = test(net, test_loader)
print(f"Final Accuracy: {accuracy:.2f}%")
torch.save(net, 'model.pth')

# TODO: Visualize 6 of the data samples from the testing dataset in a 2x3 grid
# along with the corresponding predicted digit value as a label. You can comment
# out the training step (and saving the model) above if you implement this after
# training, so that you do not have to retrain.
net = torch.load('model.pth')
