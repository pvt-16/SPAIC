## General Steps to make a network in PyTorch:
* Make a forward pass through the network 
* Use the network output to calculate the loss
* Perform a backward pass through the network to calculate the gradients
* Take a step with the optimizer to update the weights

## Training the network
Put the algorithm into a loop. one pass through the entire dataset is called an _epoch_


## MNIST- Fashion Dataset

Build a similar model to the MNIST dataset (Numbers) for classification of clothes.

### Download and load the training data

`trainset = datasets.FashionMNIST('~/.pytorch/Fashion-MNIST_data/', download=True, train=True, transform=transform)`

`trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)`

Code at: https://github.com/pvt-16/SPAIC/blob/master/Lesson%202/fashion_mnist.py

In case you want to test the model against the data and measure the accuracy of the model, download and load test data with:

`testset = datasets.FashionMNIST('~/.pytorch/Fashion-MNIST_data/', download=True, train=False, transform=transform)`

`testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)`

And using the test data set with:
https://gist.github.com/pvt-16/14ccfedc662d8cefd29b19ec758d6e5d
