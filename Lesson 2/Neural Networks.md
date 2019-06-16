# Building Neural Networks

## Example problem: Identifying text in an image
**How?** 
Given an image, out network must identify the number in the image. Range of the number can be 0-9.
We do this by training our model on MNIST training data with images and labels. The output of the network will be a probability distribution. The label with the highest probability is the idenntified number.

### Using MNIST database
 Greyscale handwritten digits. Each image is 28x28 pixels
 
 Getting dataset - through **torchvision** package and create training and test datasets 

`from torchvision import datasets` -> use this to download MNIST dataset

After downloading the dataset, you need to load the training data.
using torch, we load the data into the variable trainloader.
`trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)`

Here, batch_size = 64 => 64 images will be loaded in one iteration from the data loader and passed through our network.
And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again.

We are building a *fully-connected* or *dense networks*. Each unit in one layer is connected to each unit in the next layer. In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples).

### Preparing images for network

Our images in MNIST dataset are 28x28 => 2D tensors. We must convert them into 1D vectors by 'flattening'.

`flattened_images= images.view(images.shape[0], -1)`

After flattening, we pass the images through our model. Code at: https://github.com/pvt-16/SPAIC/blob/master/Lesson%202/neural_networks%20-%20basic.py

WE have used softmax function in the code. Explaination below.

### Output- Probability Distribution output

Using **softmax** activation function - Normalizes values

<img src="https://i.stack.imgur.com/iP8Du.png" alt="Softmax" style="float: left; margin-right: 10px;" />
 takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities
 
Using pytorch nn module: `torch.nn.Softmax(dim=1)`

OR it can be implemented as follows: 

`def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)
`

torch.sum(torch.exp(x), dim=1) -> gives us a tensor that is a vector of 64 elements (in this case)
Directly dividing will give a 64x64 tensor. Hence, We are reshaping it to give a 64x1 tensor.

### Using Pytorch nn module
Let's redefine the model using Pytorch nn module.

It is mandatory to inherit from `nn.Module` when you're creating a class for your network.

`import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
`
In the code, we have, `self.hidden = nn.Linear(784, 256)`

This line creates a module for a linear transformation, 𝑥𝐖+𝑏, with 784 inputs and 256 outputs and assigns it to self.hidden. The module automatically creates the weight and bias tensors. It is accessible after the network is created (as in, an object of the Network class is created)

Here, we are using a new **sigmoid** activation function. 

### Activation functions

In general, any function can be used as an activation function. The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. 

Types of activation functions discussed
1. sigmoid
2. Hyperbolic tangent
3. Rectified Linear Unit (ReLU) - most exclusively used for hidden layers

### Network with hidden layers

Network class is re-written as Network_2_layers.
Code at: https://github.com/pvt-16/SPAIC/blob/master/Lesson%202/nn_with_hidden_layer.py

**Why hidden layer?**

Create the Network. `model = Network_2_layers()`

After creating the network, we initialize the weights and biases.

Set biases to all zeros and sample from random normal with standard dev = 0.01. weights and bias customization - doing only for hidden layer 1 here
 
`model.hidden_layer_1.bias.data.fill_(0)
 model.hidden_layer_1.weight.data.normal_(std=0.01)
`

Here we are adding a helper module- method name view_classify - Developed by Udacity

### Forward pass - getting output with image inputs

Now, just pass an image and run the network. This is the *forward pass*.

Code at: https://github.com/pvt-16/SPAIC/blob/master/Lesson%202/nn_hidden_layer_forward_pass.py

As you can see above, our network has basically no idea what this digit is. It's because we haven't trained it yet, all the weights are random!

There is a gap between the correct labels and the 'predicted' label above. To measure how far our network's prediction is from the correct label, we use **loss function** 

### Loss
Loss depends on output. Output depends on weights (and bias). So to minimize the loss, we must change weights.
For the network to change/adjust weights, we use **Gradient descent**.
Gradient is the slope of the loss function- points to the fastest descent.

Algorithm for calculation loss: **Backpropogation algorithm**

This algo is basically a chain of changes. In forward pass, the output is used to calculate the loss and adjust the weights. The input is passed through the network again to see an output with lesser loss. In *backward pass*, after we get the output, we get the derivatives for the functions. As we propogate backwards, we multiply the incoming gradient with the function's gradient and so on until we reach the initial weights. Here, we can calculate the gradient of loss w.r.t. the weights. Subtract the gradient and start process again. 

Update our weights using this gradient with some learning rate 𝛼. 
𝑊′1=𝑊1−𝛼 * ( ∂ℓ/ ∂𝑊1 )

### Calculating losses in PyTorch

The nn module has methods for calculating losses. 

1. Cross-entropy loss - `nn.CrossEntropyLoss`
  - Classification problems.
 
Loss functions take 2 parameters - model outputs (logits) and correct labels
  
Typically, we assign loss function to variable `criterion`. 
Inputs to the loss function is the same as inputs to the softmax function. These inputs are supposed to be the scores for each class; not probabilities, i.e, the results from our model. They are called *logits*.

**Logits** - The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

Hence, we don't need to define a softmax function separetely. The criterion (loss function defined in pytorch and assigned here to this) combines loss function and softmax function.

Code at: https://github.com/pvt-16/SPAIC/blob/master/Lesson%202/nn_hl_forwardpass_withloss.py
PS: In code, we are re-writing the model using **nn.Sequential**

Throguh experince it is realised that, we should use log-softmax function (`nn.LogSoftmax` or `nn.functional.log_softmax`) instead of normal softmax to calculate outputs. Then we can get the probabilities by taking out the exponential using `torch.exp(output)`.

Since our network passes the inputs through softmax function, the output is not logits but probabilities itself. Hence, we will use a different loss function -

2. Negative log likelihood loss.  - `nn.NLLLoss`

### Using losses to perform backpropogation

#### Autograd module in pytoch

Autograd module automatically calculates gradients of tensors. It keeps a track of all the operations performed on the tensors and we can use this to perform backpropogation and calculate gradients wrt to the loss. 
To use autograd on a specific tensor,  set `requires_grad = True` on a tensor. 

Turn off gradients for a block of code with the `with torch.no_grad() `

Turn on or off gradients altogether with `torch.set_grad_enabled(True|False).` on the tensor.

`z.backward()` - The gradients are computed automatically with this method. It is wrt the output of the operation. This does a backward pass through the operations that created `z`. (actions performed on a tensor `y` that create another tensor variable) `z`.

`z.grad` -  attribute that holds the gradient for this tensor.

`z.grad_fn` - To see the operations performed. Each tensor has a `.grad_fn` attribute that references a Function that has created the Tensor.

Autograd: https://pytorch.org/docs/stable/autograd.html
