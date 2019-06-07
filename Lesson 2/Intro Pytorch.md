## Single Layer Neural Networks

Deep Learning is based on artificial neural networks which have been around in some form since the late 1950s. The networks are built from individual parts approximating neurons, typically called units or simply "neurons." Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.

**autograd** - to calculate gradients for training neural networks. Autograd, in my opinion, is amazing. It does all the work of backpropagation for you by calculating the gradients at each operation in the network which you can then use to update the network weights.

**transfer learning** -  use pre-trained networks to improve the performance of your classifier

**Tensors** - the main data structure of PyTorch. 
- vector is an instance of a tensor
1D tensor = row vector, 2D tensor = matrix.

Matrix multiplications are more efficient
Use torch.mm or torch.matmul
torch.matmul -> supports broadcasting. Accepts matrices with differnet sizes. Can give unexcepected output.
torch.mm -> more strict about matrix sizes

Linear algebra matrix multiplication rule -
  `T1 = [a x b]`
  `T2 =[b x a]` where a and b are matrix sizes
  
 To see the shape of a tensor called tensor, use tensor.shape.
 
 To change the shape, there are 3 methods:
  .reshape(), .resize_(), and .view().
  
`reshape()` - can copy data into a clone and return the new Tensor - uses extra memory

`resize` - same memory but it can cut-off or add data if the number of elements is incorrect.

`view` - same memory. changes the tensor shape only. throws error is number of elements is incorrect.


## Multilayer Neural Networks
 Individual units are stacked into layers and stacks of layers, into a network of neurons. The output of one layer of neurons becomes the input for the next layer. With multiple input units and output units, we now need to express the weights as a matrix.

input layer -> hidden layers=>> output layer

The number of hidden units a parameter of the network, often called a hyperparameter to differentiate it from the weights and biases parameters.

## Numpy to Torch and back

To create a tensor from a Numpy array, use `torch.from_numpy()`

To convert a tensor to a Numpy array, use the `.numpy()` method.

The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.
