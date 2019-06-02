## Single Layer Neural Networks

Deep Learning is based on artificial neural networks which have been around in some form since the late 1950s. The networks are built from individual parts approximating neurons, typically called units or simply "neurons." Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.

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
