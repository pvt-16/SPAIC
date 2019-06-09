## Identifying text in an image

### Using MNIST database
 Greyscale handwritten digits. Each image is 28x28 pixels
 
 Getting dataset - through **torchvision** package and create training and test datasets 

`from torchvision import datasets` -> use this to download MNIST dataset

After downloading the dataset, you need to load the training data.
using torch, we load the data.
`trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)`

Here, batch_size = 64 => 64 images will be loaded in one iteration from the data loader and passed through our network.

And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again.

### Probability Distribution output

Using **softmax** function - Normalizes values

<img src="https://i.stack.imgur.com/iP8Du.png" alt="Softmax" style="float: left; margin-right: 10px;" />
 takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities
 
Using pytorch nn module: `torch.nn.Softmax(dim=1)`

OR it can be implemented as follows:

`def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)
`

torch.sum(torch.exp(x), dim=1) -> gives us a tensor that is a vector of 64 elements (in this case)
Directly dividing will give a 64x64 tensor. Hence, We are reshaping it to give a 64x1 tensor.

### Pytorch nn module

Types of activation function
1. sigmoid
2. Hyperbolic tangent
3. Rectified Linear Unit (ReLU)

To measure how far our network's prediction is from the correct label, we use **loss function** 

Loss depends on output. Output depends on weights (and bias). So the minimize loss, we must change weights.
For the network to adjust weights, we use **Gradient descent**.
Gradient is the slope of the loss function- points to the fastest descent.

#### Backpropogation algorithm
Chain of changes.

In forward pass, the output is used to calculate the loss and adjust the weights. The input is passed through the network again to see an output with lesser loss. In backward pass, after we get the output, we get the derivatives for the functions. As we propogate backwards, we multiply the incoming gradient with the function's gradient and so on until we reach the initial weights. Here, we can calculate the gradient of loss w.r.t. the weights. Subtract the gradient and start process again.

Update our weights using this gradient with some learning rate 𝛼. 

𝑊′1=𝑊1−𝛼 * ( ∂ℓ/ ∂𝑊1 )

### Calculating losses in PyTorch

helper module - Developed by Udacity

Using nn.Sequential
