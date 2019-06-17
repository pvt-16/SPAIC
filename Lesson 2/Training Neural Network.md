## General Steps to make a network in PyTorch:
* Make a forward pass through the network 
* Use the network output to calculate the loss
* Perform a backward pass through the network to calculate the gradients
* Take a step with the optimizer to update the weights

## Training the network
Put the algorithm into a loop. one pass through the entire dataset is called an _epoch_
